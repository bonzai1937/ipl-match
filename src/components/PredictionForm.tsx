import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';

// Team and player data
const teams = [
  'Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore',
  'Kolkata Knight Riders', 'Delhi Capitals', 'Punjab Kings',
  'Rajasthan Royals', 'Sunrisers Hyderabad', 'Lucknow Super Giants',
  'Gujarat Titans'
];

const players = {
  'Mumbai Indians': ['Rohit Sharma', 'Ishan Kishan', 'Suryakumar Yadav', 'Tilak Varma', 'Tim David'],
  'Chennai Super Kings': ['Ruturaj Gaikwad', 'Devon Conway', 'Ajinkya Rahane', 'Shivam Dube', 'Moeen Ali'],
  'Royal Challengers Bangalore': ['Virat Kohli', 'Faf du Plessis', 'Glenn Maxwell', 'Dinesh Karthik', 'Wanindu Hasaranga'],
  'Kolkata Knight Riders': ['Shreyas Iyer', 'Nitish Rana', 'Andre Russell', 'Sunil Narine', 'Rinku Singh'],
  'Delhi Capitals': ['David Warner', 'Prithvi Shaw', 'Rishabh Pant', 'Axar Patel', 'Kuldeep Yadav'],
  'Punjab Kings': ['Shikhar Dhawan', 'Liam Livingstone', 'Jitesh Sharma', 'Sam Curran', 'Kagiso Rabada'],
  'Rajasthan Royals': ['Jos Buttler', 'Sanju Samson', 'Yashasvi Jaiswal', 'Shimron Hetmyer', 'Yuzvendra Chahal'],
  'Sunrisers Hyderabad': ['Kane Williamson', 'Aiden Markram', 'Rahul Tripathi', 'Nicholas Pooran', 'Bhuvneshwar Kumar'],
  'Lucknow Super Giants': ['KL Rahul', 'Quinton de Kock', 'Marcus Stoinis', 'Deepak Hooda', 'Ravi Bishnoi'],
  'Gujarat Titans': ['Hardik Pandya', 'Shubman Gill', 'David Miller', 'Rashid Khan', 'Mohammed Shami']
};

const venues = [
  'MA Chidambaram Stadium', 'Wankhede Stadium', 'Eden Gardens',
  'Narendra Modi Stadium', 'Rajiv Gandhi International Stadium',
  'M Chinnaswamy Stadium', 'Arun Jaitley Stadium', 'PCA Stadium'
];

const pitchTypes = [
  'Batting Friendly', 'Bowling Friendly', 'Neutral', 'Spin Friendly', 'Pace Friendly'
];

interface FormData {
  battingTeam: string;
  bowlingTeam: string;
  batsman: string;
  bowler: string;
  venue: string;
  overs: string;
  wickets: string;
  runs: string;
  pitch: string;
}

interface PredictionResponse {
  success: boolean;
  predicted_score?: number;
  error?: string;
}

const PredictionForm = () => {
  const [formData, setFormData] = useState<FormData>({
    battingTeam: '',
    bowlingTeam: '',
    batsman: '',
    bowler: '',
    venue: '',
    overs: '',
    wickets: '',
    runs: '',
    pitch: ''
  });
  
  const [availableBatsmen, setAvailableBatsmen] = useState<string[]>([]);
  const [availableBowlers, setAvailableBowlers] = useState<string[]>([]);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // Update available players when teams change
  useEffect(() => {
    if (formData.battingTeam) {
      setAvailableBatsmen(players[formData.battingTeam as keyof typeof players] || []);
      setFormData(prev => ({ ...prev, batsman: '' }));
    }
    if (formData.bowlingTeam) {
      setAvailableBowlers(players[formData.bowlingTeam as keyof typeof players] || []);
      setFormData(prev => ({ ...prev, bowler: '' }));
    }
  }, [formData.battingTeam, formData.bowlingTeam]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    setPrediction(null);
    
    try {
      // Validate all required fields
      const requiredFields = ['battingTeam', 'bowlingTeam', 'batsman', 'bowler', 
                            'venue', 'overs', 'wickets', 'runs', 'pitch'];
      const missingFields = requiredFields.filter(field => !formData[field as keyof FormData]);

      if (missingFields.length > 0) {
        setError(`Please fill all required fields: ${missingFields.join(', ')}`);
        return;
      }

      // Convert and validate numerical inputs
      const oversValue = parseFloat(formData.overs);
      const wicketsValue = parseInt(formData.wickets);
      const runsValue = parseInt(formData.runs);
      
      if (isNaN(oversValue) || oversValue < 0 || oversValue > 20) {
        setError("Please enter a valid overs value (0-20)");
        return;
      }
      
      if (isNaN(wicketsValue) || wicketsValue < 0 || wicketsValue > 10) {
        setError("Please enter a valid wickets value (0-10)");
        return;
      }
      
      if (isNaN(runsValue) || runsValue < 0) {
        setError("Please enter a valid runs value");
        return;
      }

      // Calculate derived features
      const runRate = runsValue / Math.max(oversValue, 0.1);
      const wicketsRemaining = 10 - wicketsValue;
      const resourcesRemaining = (20 - oversValue) * (wicketsRemaining / 10);
      
      // Determine match phase
      let phase = 'Middle1';
      if (oversValue <= 6) phase = 'Powerplay';
      else if (oversValue <= 10) phase = 'Middle1';
      else if (oversValue <= 15) phase = 'Middle2';
      else phase = 'Death';

      // Prepare payload for the API
      const payload = {
        venue: formData.venue,
        batting_team: formData.battingTeam,
        bowling_team: formData.bowlingTeam,
        batter: formData.batsman,
        bowler: formData.bowler,
        over: oversValue,
        cumulative_runs: runsValue,
        cumulative_wickets: wicketsValue,
        phase: phase,
        pitch_type: formData.pitch,
        run_rate: runRate,
        wickets_remaining: wicketsRemaining,
        resources_remaining: resourcesRemaining,
        is_death_over: phase === 'Death' ? 1 : 0,
        strike_rate: runRate * 100,
        batter_avg: 25,
        bowler_avg: 7,
        partnership_runs: Math.floor(runsValue * 0.3),
        partnership_balls: Math.floor(oversValue * 6 * 0.3),
        bowler_balls_bowled: Math.floor(oversValue * 2),
        bowler_recent_economy: 7,
        runs_last_5_overs: Math.floor(runsValue * 0.4),
        wickets_last_5_overs: Math.floor(wicketsValue * 0.3),
        required_run_rate: 8,
        run_rate_delta: runRate - 8,
        partnership_momentum: runRate * 0.8,
        batter_last5: 25,
        bowler_last5: 7,
        balls_faced_current_over: 3,
        runs_current_over: 5,
        current_over_rate: 8,
        phase_avg_runs: phase === 'Powerplay' ? 8 : phase === 'Death' ? 12 : 6
      };

      // Make API call
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData: PredictionResponse = await response.json();
        throw new Error(errorData.error || `Server error (${response.status})`);
      }

      const data: PredictionResponse = await response.json();
      
      if (!data.success || data.predicted_score === undefined) {
        throw new Error(data.error || 'Invalid prediction response');
      }

      setPrediction(data.predicted_score);
      
    } catch (error) {
      console.error('Prediction error:', error);
      setError(error instanceof Error ? error.message : "Failed to get prediction");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto space-y-6 p-4">
      {/* Header */}
      <div className="text-center py-4 bg-gradient-to-r from-blue-900 to-yellow-600 rounded-lg shadow-md">
        <h1 className="text-2xl font-bold text-white">IPL Match Score Prediction</h1>
        <p className="text-white mt-2">Enter match parameters to predict the final score</p>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Prediction Form */}
      <Card className="bg-white shadow-lg">
        <CardContent className="p-6">
          <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Team Selection */}
            <div className="space-y-2">
              <Label htmlFor="battingTeam">Batting Team *</Label>
              <Select 
                value={formData.battingTeam} 
                onValueChange={(value) => setFormData({...formData, battingTeam: value})}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select batting team" />
                </SelectTrigger>
                <SelectContent>
                  {teams.map((team) => (
                    <SelectItem key={team} value={team}>{team}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="bowlingTeam">Bowling Team *</Label>
              <Select 
                value={formData.bowlingTeam} 
                onValueChange={(value) => setFormData({...formData, bowlingTeam: value})}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select bowling team" />
                </SelectTrigger>
                <SelectContent>
                  {teams.filter(t => t !== formData.battingTeam).map((team) => (
                    <SelectItem key={team} value={team}>{team}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Player Selection */}
            <div className="space-y-2">
              <Label htmlFor="batsman">Batsman *</Label>
              <Select 
                value={formData.batsman} 
                onValueChange={(value) => setFormData({...formData, batsman: value})}
                disabled={!formData.battingTeam}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select batsman" />
                </SelectTrigger>
                <SelectContent>
                  {availableBatsmen.map((player) => (
                    <SelectItem key={player} value={player}>{player}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="bowler">Bowler *</Label>
              <Select 
                value={formData.bowler} 
                onValueChange={(value) => setFormData({...formData, bowler: value})}
                disabled={!formData.bowlingTeam}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select bowler" />
                </SelectTrigger>
                <SelectContent>
                  {availableBowlers.map((player) => (
                    <SelectItem key={player} value={player}>{player}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Match Conditions */}
            <div className="space-y-2">
              <Label htmlFor="venue">Venue *</Label>
              <Select 
                value={formData.venue} 
                onValueChange={(value) => setFormData({...formData, venue: value})}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select venue" />
                </SelectTrigger>
                <SelectContent>
                  {venues.map((venue) => (
                    <SelectItem key={venue} value={venue}>{venue}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="pitch">Pitch Type *</Label>
              <Select 
                value={formData.pitch} 
                onValueChange={(value) => setFormData({...formData, pitch: value})}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select pitch type" />
                </SelectTrigger>
                <SelectContent>
                  {pitchTypes.map((pitch) => (
                    <SelectItem key={pitch} value={pitch}>{pitch}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Match Stats */}
            <div className="space-y-2">
              <Label htmlFor="overs">Overs Completed *</Label>
              <Input
                type="number"
                id="overs"
                placeholder="e.g., 15.3"
                step="0.1"
                min="0"
                max="20"
                value={formData.overs}
                onChange={(e) => setFormData({...formData, overs: e.target.value})}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="wickets">Wickets Fallen *</Label>
              <Input
                type="number"
                id="wickets"
                placeholder="e.g., 3"
                min="0"
                max="10"
                value={formData.wickets}
                onChange={(e) => setFormData({...formData, wickets: e.target.value})}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="runs">Current Runs *</Label>
              <Input
                type="number"
                id="runs"
                placeholder="e.g., 125"
                min="0"
                value={formData.runs}
                onChange={(e) => setFormData({...formData, runs: e.target.value})}
              />
            </div>

            <div className="col-span-full">
              <Button 
                type="submit" 
                className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
                disabled={isLoading}
              >
                {isLoading ? (
                  <span className="flex items-center justify-center">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Predicting...
                  </span>
                ) : 'Predict Score'}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      {/* Prediction Result */}
      {prediction !== null && (
        <Card className="bg-white shadow-lg border-green-200">
          <CardHeader>
            <CardTitle className="text-2xl font-bold text-center">
              Predicted Final Score
            </CardTitle>
          </CardHeader>
          <CardContent className="text-center">
            <div className="text-5xl font-bold text-green-600 my-4">
              {Math.round(prediction)} runs
            </div>
            <div className="text-gray-600">
              {formData.battingTeam} predicted total
            </div>
            <div className="mt-4 text-sm text-gray-500">
              <p>Current: {formData.runs}/{formData.wickets} in {formData.overs} overs</p>
              <p>Venue: {formData.venue} ({formData.pitch})</p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default PredictionForm;