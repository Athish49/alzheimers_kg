'use client';
import { useEffect, useRef } from 'react';
import { Icon } from './Icons';

export function Composer({ value, onChange, onSubmit, onCancel, loading, inputRef }) {
  const internalRef = useRef(null);
  const ref = inputRef || internalRef;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { ref.current?.focus(); }, []);
  useEffect(() => {
    if (!ref.current) return;
    ref.current.style.height = "auto";
    ref.current.style.height = Math.min(200, ref.current.scrollHeight) + "px";
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  return (
    <div className="composer-wrap">
      <div className="composer">
        <textarea
          ref={ref}
          rows={1}
          placeholder="Ask a follow-up question..."
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={loading}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              if (!loading) onSubmit();
            }
          }}
        />
        <div className="composer-row">
          <div className="composer-hint">Enter to submit · Shift+Enter newline</div>
          {loading ? (
            <button className="composer-send stop" onClick={onCancel}>
              <Icon.Stop /> Stop
            </button>
          ) : (
            <button className="composer-send" onClick={onSubmit} disabled={!value.trim()}>
              <Icon.Send /> Send
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
