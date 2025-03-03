import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const REMOTE_IP = "192.168.1.158"

function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState('');
  const [showContext, setShowContext] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    
    setLoading(true);
    setError('');
    
    try {
      const result = await axios.post(`http://${REMOTE_IP}:8000/api/query`, {
        query: query,
        show_context: showContext,
        top_k: 5
      });
      
      setResponse(result.data);
    } catch (err) {
      console.error('Error fetching response:', err);
      setError(err.response?.data?.detail || 'An error occurred while processing your query.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>Smart Health Insights</h1>
        <p className="subtitle">Medical information powered by RAG</p>
      </header>
      
      <main>
        <form onSubmit={handleSubmit} className="query-form">
          <div className="input-group">
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a medical question..."
              rows={3}
              required
            />
            <div className="context-toggle">
              <label>
                <input
                  type="checkbox"
                  checked={showContext}
                  onChange={() => setShowContext(!showContext)}
                />
                Show source context
              </label>
            </div>
          </div>
          <button type="submit" disabled={loading}>
            {loading ? 'Processing...' : 'Ask Question'}
          </button>
        </form>
        
        {error && <div className="error-message">{error}</div>}
        
        {response && (
          <div className="response-container">
            <div className="answer">
              <h2>Answer</h2>
              <div className="answer-content">
                {response.answer.split('\n').map((paragraph, i) => (
                  <p key={i}>{paragraph}</p>
                ))}
              </div>
            </div>
            
            {response.context && (
              <div className="context">
                <h2>Sources</h2>
                {response.context.map((ctx, i) => (
                  <div key={i} className="context-item">
                    <p>{ctx}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;