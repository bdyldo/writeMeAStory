import React, { useState, useEffect, useRef } from "react";
import type { ConnectionStatus } from "../Common/types.ts";
import { io, Socket } from "socket.io-client";
import type {
  ServerToClientEvents,
  ClientToServerEvents,
} from "../Common/socket.interface.ts";

interface UseSocketIOReturn {
  connectionStatus: ConnectionStatus;
  socketRef: React.MutableRefObject<Socket | null>;
  emit: (event: string, data: any) => void;
  isConnected: boolean;
}

const useSocketIO = (
  serverUrl: string = "http://localhost:8000"
): UseSocketIOReturn => {
  const [connectionStatus, setConnectionStatus] =
    useState<ConnectionStatus>("Disconnected");
  const [isConnected, setIsConnected] = useState(false);

  // First param: receiving data interface
  // Second param: sending data interface
  const socketRef = useRef<Socket<
    ServerToClientEvents,
    ClientToServerEvents
  > | null>(null);

  // Function to connect to the Socket.IO server
  const connectSocket = (): void => {
    try {
      // Create socket connection with options
      socketRef.current = io(serverUrl, {
        // 🔧 IMPORTANT CONFIGURATION OPTIONS
        autoConnect: true, // Auto-connect on creation
        reconnection: true, // Enable auto-reconnection
        reconnectionDelay: 1000, // Wait 1s before reconnecting
        reconnectionDelayMax: 5000, // Max wait time between attempts
        reconnectionAttempts: 5, // Try 5 times then give up
        timeout: 20000, // Connection timeout
        transports: ["websocket", "polling"], // Fallback transports
      });

      // ✅ CONNECTION EVENTS
      socketRef.current.on("connect", () => {
        console.log("🟢 Socket.IO connected:", socketRef.current?.id);
        setConnectionStatus("connected");
        setIsConnected(true);
      });

      socketRef.current.on("disconnect", (reason) => {
        console.log("🔴 Socket.IO disconnected:", reason);
        setConnectionStatus("Disconnected");
        setIsConnected(false);

        // Handle different disconnect reasons
        if (reason === "io server disconnect") {
          // Server deliberately disconnected - don't auto-reconnect
          console.log("Server disconnected client - manual reconnect needed");
        }
      });

      // ❌ ERROR HANDLING
      socketRef.current.on("connect_error", (error) => {
        console.error("💥 Socket.IO connection error:", error.message);
        setConnectionStatus("error");
        setIsConnected(false);
      });

      // 🔄 RECONNECTION EVENTS
      socketRef.current.on("reconnect", (attemptNumber: number) => {
        console.log(`🔄 Reconnected after ${attemptNumber} attempts`);
        setConnectionStatus("connected");
        setIsConnected(true);
      });

      socketRef.current.on("reconnect_error", (error: any) => {
        console.error("🔄❌ Reconnection failed:", error);
        setConnectionStatus("error");
      });

      socketRef.current.on("reconnect_failed", () => {
        console.error("🔄💀 Reconnection failed permanently");
        setConnectionStatus("error");
      });
    } catch (error) {
      console.error("Failed to create Socket.IO connection:", error);
      setConnectionStatus("error");
    }
  };

  // Safe emit function with connection check
  const emit = (event: string, data: any): void => {
    if (socketRef.current?.connected) {
      console.log(`📤 Emitting ${event}:`, data);
      socketRef.current.emit(event as any, data);
    } else {
      console.error(`❌ Cannot emit ${event} - socket not connected`);
    }
  };

  useEffect(() => {
    connectSocket();

    // 🧹 CLEANUP
    return () => {
      if (socketRef.current) {
        console.log("🧹 Cleaning up Socket.IO connection");
        socketRef.current.disconnect();
        socketRef.current.removeAllListeners(); // Remove all event listeners
        socketRef.current = null;
      }
    };
  }, [serverUrl]);

  // Return a state of connection status, socket reference using socketio, emit function, and connection state
  return {
    connectionStatus,
    socketRef,
    emit,
    isConnected,
  };
};

export default useSocketIO;
