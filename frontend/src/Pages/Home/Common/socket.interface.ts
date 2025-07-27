export interface ServerToClientEvents {
  // Define what events server can send to client
  story_token: (data: { content: string }) => void;
  story_complete: () => void;
  story_error: (data: { message: string }) => void;
}

export interface ClientToServerEvents {
  // Define what events client can send to server
  generate_story: (data: {
    prompt: string;
    max_tokens: number;
    temperature: number;
  }) => void;
}
