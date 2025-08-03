export type ConnectionStatus = "Connected" | "Disconnected" | "Error";

// Used inside SettingsPanel and Home
export interface GenerationSettings {
  maxTokens: number;
  temperature: number;
  streamResponse: boolean;
}
