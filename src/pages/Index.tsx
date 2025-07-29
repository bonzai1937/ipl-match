
import React from 'react';
import TopBar from '@/components/TopBar';
import TeamLogo from '@/components/TeamLogo';
import { Button } from '@/components/ui/button';
import { Link } from 'react-router-dom';

const Index = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <TopBar />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600 mb-4">
            IPL Score Predictor
          </h1>
          <p className="text-xl text-gray-600 mb-8">
            Predict match scores using advanced analytics and machine learning
          </p>
        </div>

        {/* Featured Match */}
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-12">
          <h2 className="text-2xl font-bold text-center text-gray-800 mb-8">
            Today's Featured Match
          </h2>
          
          <div className="flex items-center justify-center space-x-12 mb-8">
            <TeamLogo teamName="Mumbai Indians" color="#004BA0" />
            <div className="text-center">
              <div className="text-4xl font-bold text-gray-600 mb-2">VS</div>
              <div className="text-sm text-gray-500">7:30 PM IST</div>
              <div className="text-sm text-gray-500">Wankhede Stadium</div>
            </div>
            <TeamLogo teamName="Chennai Super Kings" color="#F7B600" />
          </div>

          <div className="text-center">
            <Link to="/main">
              <Button className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-8 py-3 text-lg font-semibold rounded-lg">
                Predict This Match
              </Button>
            </Link>
          </div>
        </div>

        {/* Features */}
        <div className="grid md:grid-cols-3 gap-8">
          <div className="bg-white rounded-xl shadow-lg p-6 text-center">
            <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl">🏏</span>
            </div>
            <h3 className="text-xl font-semibold text-gray-800 mb-2">Advanced Analytics</h3>
            <p className="text-gray-600">Using cutting-edge ML algorithms for accurate predictions</p>
          </div>
          
          <div className="bg-white rounded-xl shadow-lg p-6 text-center">
            <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl">📊</span>
            </div>
            <h3 className="text-xl font-semibold text-gray-800 mb-2">Real-time Data</h3>
            <p className="text-gray-600">Live match data integration for precise forecasting</p>
          </div>
          
          <div className="bg-white rounded-xl shadow-lg p-6 text-center">
            <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl">🎯</span>
            </div>
            <h3 className="text-xl font-semibold text-gray-800 mb-2">High Accuracy</h3>
            <p className="text-gray-600">Proven track record with 85%+ prediction accuracy</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
