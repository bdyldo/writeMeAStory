import React, {useState, useEffect, useRef} from 'react'
import type { ConnectionStatus } from '../Common/types';

interface UseWebSocketReturn {
    connectionStatus: ConnectionStatus;
    wsRef: React.MutableRefObject<WebSocket | null>;
  }
  
  const useWebSocket = (): UseWebSocketReturn => {
    const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>("Disconnected");
    const wsRef = useRef<WebSocket | null>(null);
  
    const connectWebSocket = (): void => {
      try {
        const wsUrl = "ws://localhost:8000/ws";
        wsRef.current = new WebSocket(wsUrl);
  
        wsRef.current.onopen = () => {
          setConnectionStatus("connected");
          console.log("WebSocket connected");
        };
  
        wsRef.current.onclose = () => {
          setConnectionStatus("Disconnected");
          setTimeout(connectWebSocket, 3000);
        };
  
        wsRef.current.onerror = (error: Event) => {
          console.error("WebSocket error:", error);
          setConnectionStatus("error");
        };
      } catch (error) {
        console.error("Failed to connect WebSocket:", error);
        setConnectionStatus("error");
      }
    };
  
    useEffect(() => {
      connectWebSocket();
      return () => {
        if (wsRef.current) {
          wsRef.current.close();
        }
      };
    }, []);
  
    return { connectionStatus, wsRef };
  };

export default useWebSocket