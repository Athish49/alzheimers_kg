import React, { useEffect, useRef, useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';

const MessageList = ({ messages }) => {
  const bottomRef = useRef(null);
  const [expanded, setExpanded] = useState({});

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const toggleExpand = (id) => {
    setExpanded((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  return (
    <div className="messages-area custom-scrollbar">
      {messages.map((msg, index) => {
        const isExpanded = msg.id !== undefined && expanded[msg.id];
        const hasDetails = !!(msg.details && msg.details.context);

        return (
          <div
            key={msg.id ?? index}
            className={`message-row ${msg.role}`}
            style={{ animationDelay: `${index * 0.1}s` }}
          >
            <div
              className={`message-bubble ${msg.loading ? 'loading' : ''} ${
                isExpanded ? 'expanded' : ''
              } ${hasDetails ? 'has-toggle' : ''}`}
            >
              {msg.loading ? (
                <div className="loading-shimmer" />
              ) : (
                <>
                  <p className="message-text">{msg.content}</p>
                  {hasDetails && (
                    <button
                      className="toggle-details-btn"
                      onClick={() => toggleExpand(msg.id)}
                      aria-label={isExpanded ? 'Collapse details' : 'Expand details'}
                    >
                      {isExpanded ? (
                        <ChevronUp size={16} />
                      ) : (
                        <ChevronDown size={16} />
                      )}
                    </button>
                  )}
                  {hasDetails && isExpanded && (
                    <div className="message-details">
                      {/* Intent, notes, strategy hidden per request */}
                      {msg.details.context && (
                        <div className="detail-row">
                          <span className="detail-label">Context:</span>
                          <span className="detail-value prewrapped">
                            {msg.details.context}
                          </span>
                        </div>
                      )}
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        );
      })}
      <div ref={bottomRef} />
    </div>
  );
};

export default MessageList;
