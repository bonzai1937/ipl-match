
import React from 'react';
import TopBar from '@/components/TopBar';
import TeamLogo from '@/components/TeamLogo';
import PredictionForm from '@/components/PredictionForm';

const MainPage = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <TopBar />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Team Logos Section */}
        <div className="mb-8">
          <div className="flex items-center justify-center space-x-8 mb-8">
            <TeamLogo teamName="Mumbai Indians" color="#004BA0" />
            <div className="text-4xl font-bold text-gray-600">VS</div>
            <TeamLogo teamName="Chennai Super Kings" color="#F7B600" />
          </div>
          
          <div className="text-center">
            <h1 className="text-3xl font-bold text-gray-800 mb-2">
              IPL Match Score Prediction
            </h1>
            <p className="text-lg text-gray-600">
              Enter match parameters to predict the final score
            </p>
          </div>
        </div>

        {/* Prediction Form */}
        <PredictionForm />
      </div>
    </div>
  );
};

export default MainPage;
