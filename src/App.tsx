import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";

// Pages
import Index from "./pages/Index";
import MainPage from "./pages/MainPage";
import NewsPage from "./pages/NewsPage";
import UpcomingPage from "./pages/UpcomingPage";
import AccountPage from "./pages/AccountPage";
import NotFound from "./pages/NotFound";

// Prediction Form
import PredictionForm from "./components/PredictionForm";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/main" element={<MainPage />} />
          <Route path="/news" element={<NewsPage />} />
          <Route path="/upcoming" element={<UpcomingPage />} />
          <Route path="/account" element={<AccountPage />} />

          {/* ✅ Prediction Route */}
          <Route path="/predict" element={<PredictionForm />} />

          {/* Catch-all route for 404s */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
