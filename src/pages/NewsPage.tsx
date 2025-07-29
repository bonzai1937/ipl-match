
import React from 'react';
import TopBar from '@/components/TopBar';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

const NewsPage = () => {
  const newsItems = [
    {
      title: "IPL 2024 Season Updates",
      content: "Latest updates and announcements for the upcoming IPL season.",
      date: "2024-03-15"
    },
    {
      title: "Player Transfers and Auctions",
      content: "Keep track of all player movements and auction results.",
      date: "2024-03-10"
    },
    {
      title: "Match Schedule Released",
      content: "Complete fixture list for IPL 2024 has been announced.",
      date: "2024-03-05"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <TopBar />
      
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <h1 className="text-3xl font-bold text-gray-800 mb-8 text-center">
          IPL News & Updates
        </h1>
        
        <div className="grid gap-6">
          {newsItems.map((item, index) => (
            <Card key={index} className="bg-white shadow-lg hover:shadow-xl transition-shadow">
              <CardHeader>
                <CardTitle className="text-xl text-gray-800">{item.title}</CardTitle>
                <p className="text-sm text-gray-500">{item.date}</p>
              </CardHeader>
              <CardContent>
                <p className="text-gray-600">{item.content}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
};

export default NewsPage;
