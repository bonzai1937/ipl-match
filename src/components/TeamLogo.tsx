
import React from 'react';

interface TeamLogoProps {
  teamName: string;
  color: string;
}

const TeamLogo = ({ teamName, color }: TeamLogoProps) => {
  const getInitials = (name: string) => {
    return name.split(' ').map(word => word[0]).join('').toUpperCase();
  };

  return (
    <div className="flex flex-col items-center space-y-2">
      <div 
        className="w-20 h-20 rounded-full flex items-center justify-center text-white font-bold text-xl shadow-lg"
        style={{ backgroundColor: color }}
      >
        {getInitials(teamName)}
      </div>
      <span className="text-sm font-medium text-gray-700">{teamName}</span>
    </div>
  );
};

export default TeamLogo;
