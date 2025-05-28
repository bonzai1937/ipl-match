
import React from 'react';
import { Home, Calendar, User, Newspaper } from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';

const TopBar = () => {
  const location = useLocation();

  return (
    <div className="bg-gradient-to-r from-blue-600 via-purple-600 to-blue-800 shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center py-4">
          {/* Left side - Navigation */}
          <div className="flex items-center space-x-8">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-white rounded-full flex items-center justify-center">
                <span className="text-blue-600 font-bold text-sm">IPL</span>
              </div>
              <span className="text-white font-bold text-xl">Cricket Predictor</span>
            </div>
            
            <nav className="hidden md:flex space-x-6">
              <Link 
                to="/" 
                className={`flex items-center space-x-2 px-3 py-2 rounded-md transition-colors ${
                  location.pathname === '/' 
                    ? 'bg-white/20 text-white' 
                    : 'text-white/80 hover:text-white hover:bg-white/10'
                }`}
              >
                <Home size={20} />
                <span>Home</span>
              </Link>
              
              <Link 
                to="/news" 
                className={`flex items-center space-x-2 px-3 py-2 rounded-md transition-colors ${
                  location.pathname === '/news' 
                    ? 'bg-white/20 text-white' 
                    : 'text-white/80 hover:text-white hover:bg-white/10'
                }`}
              >
                <Newspaper size={20} />
                <span>News</span>
              </Link>
              
              <Link 
                to="/upcoming" 
                className={`flex items-center space-x-2 px-3 py-2 rounded-md transition-colors ${
                  location.pathname === '/upcoming' 
                    ? 'bg-white/20 text-white' 
                    : 'text-white/80 hover:text-white hover:bg-white/10'
                }`}
              >
                <Calendar size={20} />
                <span>Upcoming</span>
              </Link>
            </nav>
          </div>

          {/* Right side - Account */}
          <div className="flex items-center">
            <Link 
              to="/account"
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
                location.pathname === '/account'
                  ? 'bg-white/20 text-white'
                  : 'bg-white/10 hover:bg-white/20 text-white'
              }`}
            >
              <User size={20} />
              <span className="hidden sm:inline">Account</span>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TopBar;
