'use client';
import { useState } from 'react';
import { Icon } from './Icons';

export function FooterWithCopy() {
  const [copied, setCopied] = useState(false);
  const copyEmail = () => {
    navigator.clipboard.writeText('grathish49@gmail.com');
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <footer className="home-footer">
      <div>Atlas &middot; An ontology-grounded Graph RAG interface for Alzheimer&apos;s research</div>
      <div className="f-links">
        <span className="f-built-by">Built by Athish Gopal Rajesh</span>
        <span className="f-divider" />
        <a href="https://grathish.vercel.app/" target="_blank" rel="noopener noreferrer">Portfolio &amp; about</a>
        <span>&middot;</span>
        <a href="https://www.linkedin.com/in/athishgr/" target="_blank" rel="noopener noreferrer">LinkedIn</a>
        <span>&middot;</span>
        <span className="f-email-wrap">
          <a href="mailto:grathish49@gmail.com" onClick={(e) => { e.preventDefault(); copyEmail(); }}>grathish49@gmail.com</a>
          <button className="f-copy-btn" onClick={copyEmail} aria-label="Copy email address">
            {copied
              ? <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{color:'var(--fg)'}}><polyline points="2,7 5.5,11 12,3"/></svg>
              : <Icon.Copy />}
          </button>
        </span>
      </div>
    </footer>
  );
}
