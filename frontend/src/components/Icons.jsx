export const Icon = {
  Send: ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="none">
      <path d="M2 8L13.5 8M13.5 8L8.5 3M13.5 8L8.5 13" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),
  Stop: ({ size = 14 }) => (
    <svg width={size} height={size} viewBox="0 0 14 14" fill="none">
      <rect x="3" y="3" width="8" height="8" rx="1" fill="currentColor"/>
    </svg>
  ),
  Close: ({ size = 14 }) => (
    <svg width={size} height={size} viewBox="0 0 14 14" fill="none">
      <path d="M3 3L11 11M11 3L3 11" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
    </svg>
  ),
  Plus: ({ size = 14 }) => (
    <svg width={size} height={size} viewBox="0 0 14 14" fill="none">
      <path d="M7 2V12M2 7H12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
    </svg>
  ),
  Arrow: ({ size = 12 }) => (
    <svg width={size} height={size} viewBox="0 0 12 12" fill="none">
      <path d="M2 6H10M10 6L6 2M10 6L6 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),
  Copy: ({ size = 14 }) => (
    <svg width={size} height={size} viewBox="0 0 14 14" fill="none">
      <rect x="4" y="4" width="8" height="8" rx="1.5" stroke="currentColor" strokeWidth="1.3"/>
      <path d="M2.5 9V3.5C2.5 2.67 3.17 2 4 2H9" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
    </svg>
  ),
  Panel: ({ size = 14 }) => (
    <svg width={size} height={size} viewBox="0 0 14 14" fill="none">
      <rect x="2" y="2.5" width="10" height="9" rx="1.5" stroke="currentColor" strokeWidth="1.3"/>
      <path d="M8.5 2.5V11.5" stroke="currentColor" strokeWidth="1.3"/>
    </svg>
  ),
  ChevDown: ({ size = 12 }) => (
    <svg width={size} height={size} viewBox="0 0 12 12" fill="none">
      <path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),
  ChevRight: ({ size = 12 }) => (
    <svg width={size} height={size} viewBox="0 0 12 12" fill="none">
      <path d="M4.5 3L7.5 6L4.5 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),
  Refresh: ({ size = 14 }) => (
    <svg width={size} height={size} viewBox="0 0 14 14" fill="none">
      <path d="M12 7a5 5 0 1 1-1.5-3.5M12 2v3h-3" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
    </svg>
  ),
  Alert: ({ size = 14 }) => (
    <svg width={size} height={size} viewBox="0 0 14 14" fill="none">
      <path d="M7 1L13 12H1L7 1Z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round"/>
      <path d="M7 5.5V8M7 10V10.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
    </svg>
  ),
  Lock: ({ size = 14 }) => (
    <svg width={size} height={size} viewBox="0 0 14 14" fill="none">
      <rect x="2.5" y="6" width="9" height="7" rx="1.5" stroke="currentColor" strokeWidth="1.3"/>
      <path d="M4.5 6V4.5a2.5 2.5 0 0 1 5 0V6" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
      <circle cx="7" cy="9.5" r="1" fill="currentColor"/>
    </svg>
  ),
  Shield: ({ size = 14 }) => (
    <svg width={size} height={size} viewBox="0 0 14 14" fill="none">
      <path d="M7 1.5L12 3.5V7c0 2.5-2 4.5-5 5.5C4 12 2 10 2 7.5V3.5L7 1.5Z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round"/>
      <path d="M4.5 7L6.5 9L9.5 5.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),
  Clock: ({ size = 14 }) => (
    <svg width={size} height={size} viewBox="0 0 14 14" fill="none">
      <circle cx="7" cy="7" r="5.5" stroke="currentColor" strokeWidth="1.3"/>
      <path d="M7 4.5V7L9 8.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),
  User: ({ size = 14 }) => (
    <svg width={size} height={size} viewBox="0 0 14 14" fill="none">
      <circle cx="7" cy="5" r="2.5" stroke="currentColor" strokeWidth="1.3"/>
      <path d="M2 12c0-2.76 2.24-5 5-5s5 2.24 5 5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
    </svg>
  ),
  Log: ({ size = 14 }) => (
    <svg width={size} height={size} viewBox="0 0 14 14" fill="none">
      <rect x="2" y="2" width="10" height="10" rx="2" stroke="currentColor" strokeWidth="1.3"/>
      <path d="M4.5 5h5M4.5 7h5M4.5 9h3" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
    </svg>
  ),
  Logout: ({ size = 14 }) => (
    <svg width={size} height={size} viewBox="0 0 14 14" fill="none">
      <path d="M5 2H3a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h2" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
      <path d="M9.5 9.5L12 7l-2.5-2.5M12 7H6" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  ),
};
