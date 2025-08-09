import React, { useState, useEffect, useRef } from "react";
import type { ConnectionStatus } from "../../../../../common/frontend-types.ts";
import { io, Socket } from "socket.io-client";
import type {
  ServerToClientEvents,
  ClientToServerEvents,
} from "../../../../../common/socket.interface.ts";

interface UseSocketIOReturn {
  connectionStatus: ConnectionStatus;
  socketRef: React.RefObject<Socket | null>;
  emit: (event: keyof ClientToServerEvents, data: any) => void;
  isConnected: boolean;
}

let ServerUrl: string | undefined = "http://localhost:8000"; // Default server URL for Dev

// import.meta.env.PROD = true when you run npm run build
if (import.meta.env.PROD) {
  ServerUrl = undefined; // This makes Socket.IO connect to current domain
  // Check if we're running in Docker (localhost) vs real production (undefined for same-origin)
  if (
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1"
  ) {
    ServerUrl = `http://${window.location.hostname}:${
      window.location.port || "8000"
    }`;
  } else {
    ServerUrl = undefined; // This makes Socket.IO connect to current domain for real production
  }
} else {
  ServerUrl = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";
}

const useSocketIO = (
  serverUrl: string | undefined = ServerUrl
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
  // ! This function is called when the hook is initialized
  const connectSocket = (): void => {
    try {
      // Create socket connection with options
      socketRef.current = io(serverUrl, {
        autoConnect: true, // Auto-connect on creation
        reconnection: true, // Enable auto-reconnection
        reconnectionDelay: 1000, // Wait 1s before reconnecting
        reconnectionDelayMax: 5000, // Max wait time between attempts
        reconnectionAttempts: 5, // Try 5 times then give up
        timeout: 20000, // Connection timeout
        transports: ["websocket", "polling"], // Fallback transports
      });

      // ✅ CONNECTION EVENTS, emitted by server
      socketRef.current.on("connect", () => {
        console.log("🟢 Socket.IO connected:", socketRef.current?.id);
        setConnectionStatus("Connected");
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
        setConnectionStatus("Error");
        setIsConnected(false);
      });

      // 🔄 RECONNECTION EVENTS
      socketRef.current.on("reconnect", (attemptNumber: number) => {
        console.log(`🔄 Reconnected after ${attemptNumber} attempts`);
        setConnectionStatus("Connected");
        setIsConnected(true);
      });

      socketRef.current.on("reconnect_error", (error: any) => {
        console.error("🔄❌ Reconnection failed:", error);
        setConnectionStatus("Error");
      });

      socketRef.current.on("reconnect_failed", () => {
        console.error("🔄💀 Reconnection failed permanently");
        setConnectionStatus("Error");
      });
    } catch (error) {
      console.error("Failed to create Socket.IO connection:", error);
      setConnectionStatus("Error");
    }
  };

  // Safe emit function with connection check
  const emit = (event: keyof ClientToServerEvents, data: any): void => {
    if (socketRef.current?.connected) {
      console.log(`📤 Emitting ${event}:`, data);
      socketRef.current.emit(event, data);
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
