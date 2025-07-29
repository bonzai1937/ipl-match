
import React, { useState } from 'react';
import TopBar from '@/components/TopBar';
import SignInForm from '@/components/SignInForm';

const AccountPage = () => {
  const [isSignedIn, setIsSignedIn] = useState(false);

  const handleSignIn = () => {
    setIsSignedIn(true);
  };

  if (!isSignedIn) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-600 via-purple-600 to-blue-800">
        <TopBar />
        
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="flex justify-center items-center min-h-[70vh]">
            <div className="text-center">
              <h1 className="text-4xl font-bold text-white mb-8">
                Sign In to Your Account
              </h1>
              <SignInForm onSignIn={handleSignIn} />
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <TopBar />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-800 mb-8">
            Account Dashboard
          </h1>
          <p className="text-xl text-gray-600">
            Welcome! You are successfully signed in.
          </p>
        </div>
      </div>
    </div>
  );
};

export default AccountPage;
