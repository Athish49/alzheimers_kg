import React, { useState } from 'react';
import { Search, ArrowRight } from 'lucide-react';

const SearchBar = ({ onSearch, isCentered, isLoading }) => {
  const [query, setQuery] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onSearch(query);
      setQuery('');
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className={`search-form ${isCentered ? 'centered' : 'bottom'}`}
    >
      <div className="search-input-wrapper">
        <Search className="search-icon" size={20} />
        <input
          type="text"
          className="search-input"
          placeholder="Ask about Alzheimer's research..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={!query.trim() || isLoading}
          className="search-submit-btn"
        >
          <ArrowRight size={20} />
        </button>
      </div>
    </form>
  );
};

export default SearchBar;
