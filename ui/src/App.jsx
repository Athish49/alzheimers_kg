import React, { useRef, useState } from 'react';
import SearchBar from './components/SearchBar';
import MessageList from './components/MessageList';
import { Brain } from 'lucide-react';

function App() {
  const [hasStarted, setHasStarted] = useState(false);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const nextIdRef = useRef(0);

  const apiBase =
    import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

  const makeId = () => {
    const id = nextIdRef.current;
    nextIdRef.current += 1;
    return id;
  };

  const handleSearch = async (query) => {
    if (!hasStarted) {
      setHasStarted(true);
    }

    const userId = makeId();
    const loadingId = makeId();

    // Add user message + placeholder loading bubble
    setMessages((prev) => [
      ...prev,
      { id: userId, role: 'user', content: query },
      { id: loadingId, role: 'system', content: '', loading: true },
    ]);

    // Call backend
    try {
      setIsLoading(true);
      const response = await fetch(`${apiBase}/answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: query,
          return_context: true,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      const answerText = data.answer || 'No answer returned.';

      const answerMessage = {
        id: makeId(),
        role: 'system',
        content: answerText,
        details: {
          intent_type: data.intent_type,
          intent_notes: data.intent_notes,
          strategy: data.strategy,
          context: data.context,
        },
      };

      setMessages((prev) =>
        prev.map((msg) => (msg.id === loadingId ? answerMessage : msg))
      );
    } catch (error) {
      console.error('Error calling backend:', error);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === loadingId
            ? {
                ...msg,
                loading: false,
                content:
                  "Sorry, I couldn't reach the backend right now. Please try again.",
              }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleReload = () => {
    setHasStarted(false);
    setMessages([]);
    setIsLoading(false);
  };

  return (
    <div className="app-container">
      {/* Header / Logo */}
      <div className={`app-header ${hasStarted ? 'top-left' : 'centered'}`}>
        <div className="logo-container" onClick={handleReload}>
          <Brain className="logo-icon" />
          <span className="logo-text">Alzheimer's Graph RAG</span>
        </div>
      </div>

      {/* Main Content Area */}
      <main className="main-content">

        {/* Chat Area (Only visible when started) */}
        {hasStarted && (
          <MessageList messages={messages} />
        )}

        {/* Search Area */}
        <div
          className={`search-container-wrapper ${hasStarted ? 'bottom' : 'centered'
            }`}
        >
          <SearchBar
            onSearch={handleSearch}
            isCentered={!hasStarted}
            isLoading={isLoading}
          />
        </div>
      </main>
    </div>
  );
}

export default App;
