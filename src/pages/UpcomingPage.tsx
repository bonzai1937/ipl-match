
import React from 'react';
import TopBar from '@/components/TopBar';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import TeamLogo from '@/components/TeamLogo';

const UpcomingPage = () => {
  const upcomingMatches = [
    {
      team1: { name: "Mumbai Indians", color: "#004BA0" },
      team2: { name: "Chennai Super Kings", color: "#F7B600" },
      date: "2024-03-20",
      time: "7:30 PM",
      venue: "Wankhede Stadium"
    },
    {
      team1: { name: "Royal Challengers Bangalore", color: "#C8102E" },
      team2: { name: "Kolkata Knight Riders", color: "#512DA8" },
      date: "2024-03-22",
      time: "3:30 PM",
      venue: "M. Chinnaswamy Stadium"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <TopBar />
      
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <h1 className="text-3xl font-bold text-gray-800 mb-8 text-center">
          Upcoming Matches
        </h1>
        
        <div className="grid gap-6">
          {upcomingMatches.map((match, index) => (
            <Card key={index} className="bg-white shadow-lg hover:shadow-xl transition-shadow">
              <CardHeader>
                <CardTitle className="text-center text-lg text-gray-800">
                  {match.date} at {match.time}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-center space-x-8 mb-4">
                  <TeamLogo teamName={match.team1.name} color={match.team1.color} />
                  <div className="text-2xl font-bold text-gray-600">VS</div>
                  <TeamLogo teamName={match.team2.name} color={match.team2.color} />
                </div>
                <p className="text-center text-gray-600 font-medium">{match.venue}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
};

export default UpcomingPage;
