"use client";

import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { useNotification } from './NotificationProvider';

export type CallState = 'idle' | 'calling' | 'ringing' | 'connected' | 'ended';

interface CallContextProps {
    callState: CallState;
    remoteStream: MediaStream | null;
    localStream: MediaStream | null;
    incomingCallData: IncomingCallData | null;
    duration: number;
    isMuted: boolean;
    isRemoteMuted: boolean;
    toggleMute: () => void;
    acceptCall: (existingStream?: MediaStream) => void;
    declineCall: () => void;
    endCall: () => void;
    startCall: (targetUsername: string, mode?: 'audio' | 'video', contextId?: string, existingStream?: MediaStream) => void;
}

interface IncomingCallData {
    call_id: string;
    caller_username: string;
    caller_id: number;
    mode: 'audio' | 'video';
    sdp?: RTCSessionDescriptionInit;
    context_id?: string;
    caller_image_url?: string;
    caller_interests?: string[];
}

const CallContext = createContext<CallContextProps | null>(null);

export const useCall = () => {
    const context = useContext(CallContext);
    if (!context) throw new Error("useCall must be used within CallProvider");
    return context;
};

const RING_TIMEOUT_MS = 30000;

export default function CallProvider({ children }: { children: React.ReactNode }) {
    const { sendSignal } = useNotification();
    // Always call the latest sendSignal without making effects depend on it
    const sendSignalRef = useRef(sendSignal);
    sendSignalRef.current = sendSignal;
    const signal = useCallback((type: string, payload: any) => sendSignalRef.current(type, payload), []);

    const [callState, setCallState] = useState<CallState>('idle');
    const callStateRef = useRef<CallState>('idle');

    const [incomingCallData, setIncomingCallData] = useState<IncomingCallData | null>(null);
    const incomingCallDataRef = useRef<IncomingCallData | null>(null);
    const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null);
    const [localStream, setLocalStreamState] = useState<MediaStream | null>(null);
    const localStreamRef = useRef<MediaStream | null>(null);

    const [isMuted, setIsMuted] = useState(false);
    const isMutedRef = useRef(false);
    const [isRemoteMuted, setIsRemoteMuted] = useState(false);
    const [duration, setDuration] = useState(0);

    const peerConnection = useRef<RTCPeerConnection | null>(null);
    const dataChannel = useRef<RTCDataChannel | null>(null);
    const activeCallId = useRef<string | null>(null);
    const durationTimer = useRef<ReturnType<typeof setInterval> | null>(null);
    const iceCandidatesBuffer = useRef<any[]>([]);
    const ringingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    const setLocalStream = (s: MediaStream | null) => {
        localStreamRef.current = s;
        setLocalStreamState(s);
    };
    const setIncoming = (d: IncomingCallData | null) => {
        incomingCallDataRef.current = d;
        setIncomingCallData(d);
    };

    const rtcConfig: RTCConfiguration = {
        iceServers: [
            { urls: 'stun:stun.l.google.com:19302' },
            // A TURN server is still needed for calls over mobile data / strict NATs.
        ]
    };

    const startDurationTimer = () => {
        if (durationTimer.current) clearInterval(durationTimer.current);
        setDuration(0);
        durationTimer.current = setInterval(() => setDuration(d => d + 1), 1000);
    };

    // Clean up connections. Uses refs so it always sees the CURRENT stream
    // (before, it captured a stale `localStream` and the camera could stay on).
    const cleanupCall = () => {
        if (durationTimer.current) clearInterval(durationTimer.current);
        durationTimer.current = null;
        if (ringingTimerRef.current) clearTimeout(ringingTimerRef.current);
        ringingTimerRef.current = null;
        if (dataChannel.current) {
            try { dataChannel.current.close(); } catch { }
            dataChannel.current = null;
        }
        if (peerConnection.current) {
            peerConnection.current.close();
            peerConnection.current = null;
        }
        if (localStreamRef.current) {
            localStreamRef.current.getTracks().forEach(track => track.stop());
        }
        setLocalStream(null);
        setRemoteStream(null);
        setCallState('idle');
        setIncoming(null);
        activeCallId.current = null;
        setDuration(0);
        setIsMuted(false);
        isMutedRef.current = false;
        setIsRemoteMuted(false);
        iceCandidatesBuffer.current = [];
    };

    // Keep ref in sync and run the 30s ringing timeout. Depends ONLY on callState,
    // so re-renders no longer restart the timer (calls used to ring forever).
    useEffect(() => {
        callStateRef.current = callState;

        if (ringingTimerRef.current) {
            clearTimeout(ringingTimerRef.current);
            ringingTimerRef.current = null;
        }

        if (callState === 'calling' || callState === 'ringing') {
            ringingTimerRef.current = setTimeout(() => {
                console.log("[WEBRTC] 30 seconds timeout reached. Ending call automatically.");
                if (activeCallId.current) {
                    signal(callStateRef.current === 'calling' ? 'call_end' : 'call_decline', { call_id: activeCallId.current });
                }
                cleanupCall();
            }, RING_TIMEOUT_MS);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [callState]);

    const drainIceCandidates = async (pc: RTCPeerConnection) => {
        while (iceCandidatesBuffer.current.length > 0) {
            const candidate = iceCandidatesBuffer.current.shift();
            try {
                await pc.addIceCandidate(new RTCIceCandidate(candidate));
            } catch (err) {
                console.error("[WEBRTC ERROR] Error adding buffered ice candidate", err);
            }
        }
    };

    const attachDataChannel = (channel: RTCDataChannel) => {
        dataChannel.current = channel;
        channel.onmessage = (e) => {
            try {
                const data = JSON.parse(e.data);
                if (data.type === 'mute_state') setIsRemoteMuted(!!data.isMuted);
            } catch { }
        };
        channel.onopen = () => {
            // Tell the other side our current mute state as soon as the channel opens
            try { channel.send(JSON.stringify({ type: 'mute_state', isMuted: isMutedRef.current })); } catch { }
        };
    };

    const setupPeerConnection = () => {
        const pc = new RTCPeerConnection(rtcConfig);

        pc.onicecandidate = (event) => {
            if (event.candidate && activeCallId.current) {
                signal('ice_candidate', {
                    call_id: activeCallId.current,
                    candidate: event.candidate,
                });
            }
        };

        pc.ontrack = (event) => {
            setRemoteStream(event.streams[0]);
        };

        // Callee side: receive the caller's data channel. (Before, it was never stored,
        // so the callee's mute state was never sent to the caller.)
        pc.ondatachannel = (event) => attachDataChannel(event.channel);

        pc.oniceconnectionstatechange = () => {
            if (pc.iceConnectionState === 'failed') {
                console.error("[WEBRTC ERROR] ICE Connection failed. This usually means a TURN server is required for this network.");
            }
        };

        pc.onconnectionstatechange = () => {
            if (pc.connectionState === 'connected') {
                setCallState('connected');
            }
        };

        peerConnection.current = pc;
        return pc;
    };

    useEffect(() => {
        const handleIncomingCall = (e: any) => {
            const data = e.detail;
            if (!data?.call_id) return;

            if (callStateRef.current === 'calling'
                && incomingCallDataRef.current?.caller_username === data.caller_username) {
                // "Glare": we are calling them while they call us (both sides auto-dial in
                // video sessions). Resolve deterministically so exactly ONE call survives.
                let me = '';
                try { me = JSON.parse(localStorage.getItem('user') || '{}')?.username || ''; } catch { }
                if (me && me < data.caller_username) {
                    signal('call_decline', { call_id: data.call_id }); // keep our outgoing call
                    return;
                }
                // Drop our own outgoing offer (keep the camera stream) and take theirs.
                if (activeCallId.current) signal('call_end', { call_id: activeCallId.current });
                if (dataChannel.current) { try { dataChannel.current.close(); } catch { } dataChannel.current = null; }
                if (peerConnection.current) { peerConnection.current.close(); peerConnection.current = null; }
                iceCandidatesBuffer.current = [];
            } else if (callStateRef.current !== 'idle') {
                // Busy with another call: decline the new one
                signal('call_decline', { call_id: data.call_id });
                return;
            }
            setIncoming(data);
            activeCallId.current = data.call_id;
            setCallState('ringing');
        };

        const handleCallAnswered = async (e: any) => {
            const data = e.detail;
            if (data?.call_id && data.call_id !== activeCallId.current) return;
            if (peerConnection.current && data.sdp_answer) {
                try {
                    await peerConnection.current.setRemoteDescription(new RTCSessionDescription(data.sdp_answer));
                    setCallState('connected');
                    await drainIceCandidates(peerConnection.current);
                    startDurationTimer();
                } catch (err) {
                    console.error("[WEBRTC ERROR] Failed to set remote description:", err);
                }
            }
        };

        const handleIceCandidate = async (e: any) => {
            const data = e.detail;
            if (data.call_id !== activeCallId.current) return;

            if (peerConnection.current && peerConnection.current.remoteDescription) {
                try {
                    await peerConnection.current.addIceCandidate(new RTCIceCandidate(data.candidate));
                } catch (err) {
                    console.error("[WEBRTC ERROR] Error adding ice candidate", err);
                }
            } else {
                iceCandidatesBuffer.current.push(data.candidate);
            }
        };

        // Only react to signals for OUR call. (Before, any call_ended/declined event —
        // even for a different call — tore down whatever call was active.)
        const handleCallDeclined = (e: any) => {
            if (activeCallId.current && e.detail?.call_id === activeCallId.current) cleanupCall();
        };

        const handleCallEnded = (e: any) => {
            if (activeCallId.current && e.detail?.call_id === activeCallId.current) cleanupCall();
        };

        window.addEventListener('sys_incoming_call', handleIncomingCall);
        window.addEventListener('sys_call_answered', handleCallAnswered);
        window.addEventListener('sys_ice_candidate', handleIceCandidate);
        window.addEventListener('sys_call_declined', handleCallDeclined);
        window.addEventListener('sys_call_ended', handleCallEnded);

        return () => {
            window.removeEventListener('sys_incoming_call', handleIncomingCall);
            window.removeEventListener('sys_call_answered', handleCallAnswered);
            window.removeEventListener('sys_ice_candidate', handleIceCandidate);
            window.removeEventListener('sys_call_declined', handleCallDeclined);
            window.removeEventListener('sys_call_ended', handleCallEnded);
        };
        // Handlers only use refs + stable helpers, so register once.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Stop camera/mic if the provider ever unmounts mid-call
    useEffect(() => () => {
        if (activeCallId.current) signal('call_end', { call_id: activeCallId.current });
        cleanupCall();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const getMedia = async (mode: 'audio' | 'video' = 'audio') => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: true,
                video: mode === 'video'
            });
            setLocalStream(stream);
            return stream;
        } catch (err) {
            console.error("Failed to get media", err);
            return null;
        }
    };

    const startCallImpl = async (targetUsername: string, mode: 'audio' | 'video' = 'audio', contextId?: string, existingStream?: MediaStream) => {
        if (callStateRef.current !== 'idle') {
            console.warn("[WEBRTC] Already in a call; ignoring startCall");
            return;
        }
        const stream = existingStream || await getMedia(mode);
        if (existingStream) setLocalStream(existingStream);
        if (!stream) {
            console.error("[WEBRTC ERROR] Could not get media stream. Call aborted.");
            return;
        }

        const pc = setupPeerConnection();
        stream.getTracks().forEach(track => pc.addTrack(track, stream));

        // Caller side data channel (mute state in both directions)
        attachDataChannel(pc.createDataChannel('controls'));

        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        const tempCallId = crypto.randomUUID();
        activeCallId.current = tempCallId;

        setIncoming({
            call_id: tempCallId,
            caller_username: targetUsername,
            caller_id: 0,
            mode,
            context_id: contextId
        });
        setCallState('calling');

        signal('call_offer', {
            to_username: targetUsername,
            call_id: tempCallId,
            sdp: offer,
            mode,
            context_id: contextId
        });
    };

    const acceptCallImpl = async (existingStream?: MediaStream) => {
        const incoming = incomingCallDataRef.current;
        if (!incoming || !incoming.sdp) {
            console.error("[WEBRTC ERROR] No incoming call data or SDP available to accept!");
            return;
        }

        const stream = existingStream || await getMedia(incoming.mode);
        if (existingStream) setLocalStream(existingStream);
        if (!stream) {
            alert("Could not access your camera or microphone. Please ensure permissions are granted and the device is not in use by another tab.");
            return;
        }

        const pc = setupPeerConnection();
        stream.getTracks().forEach(track => pc.addTrack(track, stream));

        try {
            await pc.setRemoteDescription(new RTCSessionDescription(incoming.sdp));
            const answer = await pc.createAnswer();
            await pc.setLocalDescription(answer);
            await drainIceCandidates(pc);

            signal('call_answer', {
                call_id: incoming.call_id,
                sdp_answer: answer
            });

            setCallState('connected');
            startDurationTimer();
        } catch (err) {
            console.error("[WEBRTC ERROR] Error during acceptCall flow:", err);
        }
    };

    const declineCallImpl = () => {
        if (activeCallId.current) {
            signal('call_decline', { call_id: activeCallId.current });
        }
        cleanupCall();
    };

    const endCallImpl = () => {
        if (activeCallId.current) {
            signal('call_end', { call_id: activeCallId.current });
        }
        // Nothing to clean up if there was never a call (cheap no-op)
        if (callStateRef.current !== 'idle' || localStreamRef.current || peerConnection.current) {
            cleanupCall();
        }
    };

    const toggleMuteImpl = () => {
        const stream = localStreamRef.current;
        if (!stream) return;
        const audioTracks = stream.getAudioTracks();
        if (audioTracks.length === 0) return;
        const newMutedState = !isMutedRef.current;
        audioTracks.forEach(t => { t.enabled = !newMutedState; });
        isMutedRef.current = newMutedState;
        setIsMuted(newMutedState);
        if (dataChannel.current && dataChannel.current.readyState === 'open') {
            dataChannel.current.send(JSON.stringify({ type: 'mute_state', isMuted: newMutedState }));
        }
    };

    // Stable function identities. Pages put these in effect dependency lists; when they
    // changed on every render, the chat room's effect cleanup ran endCall() and hung up
    // calls almost immediately after they started.
    const implRef = useRef({ startCallImpl, acceptCallImpl, declineCallImpl, endCallImpl, toggleMuteImpl });
    implRef.current = { startCallImpl, acceptCallImpl, declineCallImpl, endCallImpl, toggleMuteImpl };

    const startCall = useCallback((t: string, m?: 'audio' | 'video', c?: string, s?: MediaStream) => { implRef.current.startCallImpl(t, m, c, s); }, []);
    const acceptCall = useCallback((s?: MediaStream) => { implRef.current.acceptCallImpl(s); }, []);
    const declineCall = useCallback(() => implRef.current.declineCallImpl(), []);
    const endCall = useCallback(() => implRef.current.endCallImpl(), []);
    const toggleMute = useCallback(() => implRef.current.toggleMuteImpl(), []);

    const value = useMemo(() => ({
        callState, remoteStream, localStream, incomingCallData, duration, isMuted, isRemoteMuted,
        toggleMute, acceptCall, declineCall, endCall, startCall,
    }), [callState, remoteStream, localStream, incomingCallData, duration, isMuted, isRemoteMuted,
        toggleMute, acceptCall, declineCall, endCall, startCall]);

    return (
        <CallContext.Provider value={value}>
            {children}
        </CallContext.Provider>
    );
}
