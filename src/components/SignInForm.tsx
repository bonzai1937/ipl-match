
import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

interface SignInFormProps {
  onSignIn: () => void;
}

const SignInForm = ({ onSignIn }: SignInFormProps) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSignIn();
  };

  return (
    <Card className="w-full max-w-md mx-auto bg-white/95 backdrop-blur-sm shadow-xl">
      <CardHeader className="text-center">
        <CardTitle className="text-2xl font-bold text-gray-800">Welcome Back</CardTitle>
        <CardDescription className="text-gray-600">
          Sign in to predict IPL match scores
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <Input
              type="email"
              placeholder="Email address"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full"
              required
            />
          </div>
          <div>
            <Input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full"
              required
            />
          </div>
          <Button type="submit" className="w-full bg-blue-600 hover:bg-blue-700">
            Sign In
          </Button>
        </form>
        <div className="mt-4 text-center">
          <p className="text-sm text-gray-600">
            Don't have an account?{' '}
            <button className="text-blue-600 hover:text-blue-700 font-medium">
              Sign up
            </button>
          </p>
        </div>
      </CardContent>
    </Card>
  );
};

export default SignInForm;
