from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from django.conf import settings
from .models import Payment, Subscription, SubscriptionPlan, GemLedger
import razorpay
from django.utils import timezone
from datetime import timedelta

# Razorpay Client
razorpay_client = razorpay.Client(auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET))

class PlanListView(APIView):
    permission_classes = [permissions.AllowAny]
    def get(self, request):
        plans = SubscriptionPlan.objects.filter(is_active=True).order_by('price_inr')
        data = [{
            "id": p.id,
            "name": p.name,
            "price": float(p.price_inr),
            "duration_days": p.duration_days,
            "features": p.features
        } for p in plans]
        return Response(data)

class CreateOrderView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        user = request.user
        product_type = request.data.get('product_type') # 'subscription'
        item_id = request.data.get('item_id')

        if product_type == 'subscription':
            try:
                item = SubscriptionPlan.objects.get(id=item_id, is_active=True)
                amount = int(item.price_inr * 100)
            except (SubscriptionPlan.DoesNotExist, ValueError, TypeError):
                return Response({"error": "Plan not found"}, status=status.HTTP_404_NOT_FOUND)
        else:
            # SECURITY: gem purchases were priced by the client ("amount") and fulfilled with a
            # client-sent "gems_amount" — i.e. pay ₹1, get any number of gems. Disabled until
            # gem packs are priced server-side (the frontend does not use this path today).
            return Response({"error": "Invalid product type"}, status=status.HTTP_400_BAD_REQUEST)

        # Create Razorpay Order
        payment_data = {
            "amount": amount,
            "currency": "INR",
            "receipt": f"receipt_{user.id}_{product_type}_{item.id}",
            "notes": {"user_id": str(user.id), "plan_id": str(item.id)},
        }

        try:
            order = razorpay_client.order.create(data=payment_data)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Create DB Payment record — the plan is fixed here, server-side.
        payment = Payment.objects.create(
            user=user,
            amount=amount / 100,
            product_type=product_type,
            status='pending',
            razorpay_order_id=order['id'],
            razorpay_payment_id=order['id'],  # legacy: kept for old rows / admin views
            plan=item,
        )
        
        return Response({
            "order_id": order['id'],
            "amount": amount,
            "currency": order['currency'],
            "payment_id": payment.id
        })

class VerifyPaymentView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        razorpay_order_id = request.data.get('razorpay_order_id')
        razorpay_payment_id = request.data.get('razorpay_payment_id')
        razorpay_signature = request.data.get('razorpay_signature')
        if not (razorpay_order_id and razorpay_payment_id and razorpay_signature):
            return Response({"error": "Missing payment details"}, status=status.HTTP_400_BAD_REQUEST)

        # Verify Signature FIRST (proves Razorpay really captured this order/payment pair)
        try:
            razorpay_client.utility.verify_payment_signature({
                'razorpay_order_id': razorpay_order_id,
                'razorpay_payment_id': razorpay_payment_id,
                'razorpay_signature': razorpay_signature
            })
        except razorpay.errors.SignatureVerificationError:
            Payment.objects.filter(razorpay_order_id=razorpay_order_id, user=request.user,
                                   status='pending').update(status='failed')
            return Response({"error": "Signature verification failed"}, status=status.HTTP_400_BAD_REQUEST)

        from django.db import transaction
        with transaction.atomic():
            # SECURITY: lock the row and only fulfil once; the order must belong to the caller.
            payment = (Payment.objects.select_for_update()
                       .filter(razorpay_order_id=razorpay_order_id, user=request.user)
                       .first())
            if payment is None:
                return Response({"error": "Payment not found"}, status=status.HTTP_404_NOT_FOUND)
            if payment.status == 'success':
                return Response({"status": "success", "message": "Payment already verified"})
            if payment.status != 'pending':
                return Response({"error": "Payment is not pending"}, status=status.HTTP_400_BAD_REQUEST)

            # Payment is successful
            payment.status = 'success'
            payment.razorpay_payment_id = razorpay_payment_id
            payment.save()

            # Fulfill Order — using ONLY what was stored at order creation (ignore client item_id)
            if payment.product_type == 'subscription' and payment.plan_id:
                plan = payment.plan
                Subscription.objects.update_or_create(
                    user=payment.user,
                    defaults={
                        'plan': plan,
                        'status': 'active',
                        'next_billing_date': timezone.now() + timedelta(days=plan.duration_days)
                    }
                )

                # Sync with the user's master profile to immediately apply quotas
                profile = payment.user.profile

                # Determine tier (map plan name loosely to SUBSCRIPTION_TIER_CHOICES)
                plan_name_lower = plan.name.lower()
                if 'week' in plan_name_lower:
                    profile.subscription_tier = 'weekly'
                else:
                    profile.subscription_tier = 'monthly'

                profile.subscription_expiry = timezone.now() + timedelta(days=plan.duration_days)

                # Reset quotas so they immediately benefit from premium limits
                profile.daily_ai_llm_searches = 0
                profile.daily_roulette_searches = 0
                profile.save()

        return Response({"status": "success", "message": "Payment verified and order fulfilled"})
