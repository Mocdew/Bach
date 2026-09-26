import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import "./styles.css";

const client = new QueryClient({
  defaultOptions: {
    queries: {
      retry: (n, e) => n < 2 && !(e instanceof Error && "status" in e && (e as { status: number }).status < 500),
      refetchOnWindowFocus: true,
    },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={client}><App /></QueryClientProvider>
  </StrictMode>,
);
