import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import './styles/tokens.css'
import './styles/app.css'
import App from './App.jsx'
import { HomePage } from './pages/HomePage.jsx'
import { DemoPage } from './pages/DemoPage.jsx'
import { WorkspacePage } from './pages/WorkspacePage.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/app" element={<App />} />
        <Route path="/demo" element={<DemoPage />} />
        <Route path="/workspace" element={<WorkspacePage />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
