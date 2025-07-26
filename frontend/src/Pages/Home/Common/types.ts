export type ConnectionStatus = "connected" | "Disconnected" | "error";

export interface GenerationSettings {
  maxTokens: number;
  temperature: number;
  streamResponse: boolean;
}
