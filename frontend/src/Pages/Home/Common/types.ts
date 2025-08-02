export type ConnectionStatus = "connected" | "Disconnected" | "error";

// Used inside SettingsPanel and Home
export interface GenerationSettings {
  maxTokens: number;
  temperature: number;
  streamResponse: boolean;
}
