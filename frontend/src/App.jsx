import { useState, useEffect, useRef, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { Icon } from './components/Icons';
import { Landing } from './components/Landing';
import { Composer } from './components/Composer';
import { AssistantMessage, LoadingMessage, ErrorMessage } from './components/Messages';
import { ContextPanel } from './components/ContextPanel';
import { queryAnswer } from './api';

function App() {
  const [hasStarted, setHasStarted] = useState(false);
  const [messages, setMessages] = useState([]);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingPhase, setLoadingPhase] = useState(0);
  const [panelOpen, setPanelOpen] = useState(false);
  const [pinnedMessageId, setPinnedMessageId] = useState(null);
  const [composerVal, setComposerVal] = useState("");
  const [sessionId] = useState(() => Math.random().toString(36).slice(2, 8).toUpperCase());
  const abortRef = useRef(null);
  const threadRef = useRef(null);
  const composerRef = useRef(null);

  useEffect(() => {
    if (threadRef.current) {
      threadRef.current.scrollTop = threadRef.current.scrollHeight;
    }
  }, [messages, loading]);

  const runQuery = useCallback(async (q) => {
    const userMsg = { id: "u-" + Date.now(), role: "user", text: q };
    setMessages(prev => [...prev, userMsg]);
    setHasStarted(true);
    setLoading(true);
    setLoadingPhase(0);
    setComposerVal("");

    const abortController = new AbortController();
    abortRef.current = abortController;

    const t1 = setTimeout(() => setLoadingPhase(1), 500);
    const t2 = setTimeout(() => setLoadingPhase(2), 1100);

    try {
      const resp = await queryAnswer(q, { history, signal: abortController.signal });
      clearTimeout(t1);
      clearTimeout(t2);

      const id = "a-" + Date.now();
      const aMsg = {
        id,
        role: "assistant",
        question: q,
        answer: resp.answer,
        intent_type: resp.intent_type,
        intent_notes: resp.intent_notes,
        strategy: resp.strategy,
        context: resp.evidence || null,
        followups: generateFollowups(resp.intent_type),
      };
      setMessages(prev => [...prev, aMsg]);
      setHistory(prev => [
        ...prev,
        { role: "user", content: q },
        { role: "assistant", content: resp.answer },
      ]);
      setLoading(false);

      if (resp.evidence) {
        setPanelOpen(true);
        setPinnedMessageId(id);
      }
    } catch (err) {
      clearTimeout(t1);
      clearTimeout(t2);
      if (err.name === "AbortError") {
        setLoading(false);
        return;
      }
      const id = "e-" + Date.now();
      setMessages(prev => [...prev, {
        id,
        role: "error",
        question: q,
        title: "Request failed",
        body: err.message || "Could not reach the API. Make sure the backend is running.",
      }]);
      setLoading(false);
    }
  }, []);

  const cancel = () => {
    abortRef.current?.abort();
    setLoading(false);
  };

  const handleSubmit = (text) => {
    if (!text.trim() || loading) return;
    runQuery(text.trim());
  };

  const newSession = () => {
    setMessages([]);
    setHistory([]);
    setHasStarted(false);
    setPanelOpen(false);
    setPinnedMessageId(null);
    setComposerVal("");
    setLoading(false);
  };

  const pinnedMessage = messages.find(m => m.id === pinnedMessageId) || null;

  return (
    <>
      <header className="app-header">
        <div style={{ display: "flex", alignItems: "center" }}>
          <Link className="app-brand" to="/">
            <span className="brand-mark">A</span>
            <span>Atlas</span>
          </Link>
          <span className="app-session">session · {sessionId}</span>
        </div>
        <div className="header-actions">
          <button
            className={"icon-btn " + (panelOpen ? "active" : "")}
            onClick={() => setPanelOpen(o => !o)}
            title={panelOpen ? "Hide evidence panel" : "Show evidence panel"}
            disabled={!hasStarted}
          >
            <Icon.Panel />
          </button>
          <button className="hdr-btn" onClick={newSession} disabled={!hasStarted}>
            <Icon.Plus /> New session
          </button>
        </div>
      </header>

      <main className={"app-main " + (hasStarted && panelOpen ? "with-panel" : "")}>
        <div className="chat-col">
          {!hasStarted ? (
            <Landing onSubmit={handleSubmit} onPickCategory={(q) => runQuery(q)} />
          ) : (
            <>
              <div className="thread" ref={threadRef}>
                <div className="thread-inner">
                  {messages.map((m) => {
                    if (m.role === "user") return (
                      <div key={m.id} className="msg msg-user">{m.text}</div>
                    );
                    if (m.role === "assistant") return (
                      <AssistantMessage
                        key={m.id}
                        msg={m}
                        isActive={m.id === pinnedMessageId && panelOpen}
                        onViewEvidence={() => { setPanelOpen(true); setPinnedMessageId(m.id); }}
                        onFollowup={(q) => runQuery(q)}
                      />
                    );
                    if (m.role === "error") return (
                      <ErrorMessage key={m.id} msg={m} onRetry={() => runQuery(m.question)} />
                    );
                    return null;
                  })}
                  {loading && <LoadingMessage phase={loadingPhase} />}
                </div>
              </div>
              <Composer
                value={composerVal}
                onChange={setComposerVal}
                onSubmit={() => handleSubmit(composerVal)}
                onCancel={cancel}
                loading={loading}
                inputRef={composerRef}
              />
            </>
          )}
        </div>

        {hasStarted && panelOpen && (
          <aside className="ctx-col">
            <ContextPanel
              key={pinnedMessageId}
              message={pinnedMessage}
              pinnedQuestion={pinnedMessage?.question || ""}
              onClose={() => setPanelOpen(false)}
            />
          </aside>
        )}
      </main>
    </>
  );
}

function generateFollowups(intentType) {
  const followupMap = {
    BIOMARKER: [
      "Which of these biomarkers are detectable in plasma?",
      "What is the clinical significance of p-tau181?",
      "Which biomarkers track disease progression?",
    ],
    DRUG_TRIAL: [
      "What pathways does lecanemab target?",
      "Which Phase 3 trials are actively recruiting?",
      "What are the ARIA rates across approved antibodies?",
    ],
    PHENOTYPE: [
      "At what stage does agitation typically appear?",
      "Which phenotypes differ between early and late onset AD?",
      "What are the earliest prodromal symptoms?",
    ],
    PATHWAY: [
      "How does donanemab's pathway profile compare?",
      "Which drugs also target microglial phagocytosis?",
      "What tau-related pathways are active in AD?",
    ],
    GENE_PROTEIN: [
      "What protein does APOE encode and which pathways does it affect?",
      "Which TREM2 variants raise AD risk?",
      "How do early-onset AD genes differ mechanistically?",
    ],
    GENERAL_AD: [
      "What are the main disease mechanisms in AD?",
      "How have recent FDA approvals changed the landscape?",
      "What unanswered questions remain?",
    ],
  };
  return followupMap[intentType] || followupMap.GENERAL_AD;
}

export default App;
