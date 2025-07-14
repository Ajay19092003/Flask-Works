const express = require('express');
const cors = require('cors');
const app = express();
const PORT = 3001;

app.use(cors());
app.use(express.json());

app.get('/', (req, res) => {
  res.json({ 
    message: '🚑 Disaster Management API is running!',
    version: '1.0.0',
    status: 'healthy'
  });
});

app.get('/api/hospitals', (req, res) => {
  res.json([
    {
      id: 'hosp-1',
      name: 'City General Hospital',
      address: '123 Main St, City Center',
      lat: 40.7128,
      lng: -74.0060,
      contact_phone: '+1-555-0101',
      specializations: 'Emergency,Cardiology,Trauma'
    },
    {
      id: 'hosp-2',
      name: 'Metropolitan Medical Center',
      address: '456 Health Ave, Downtown',
      lat: 40.7589,
      lng: -73.9851,
      contact_phone: '+1-555-0102',
      specializations: 'Pediatrics,Emergency,Neurology'
    }
  ]);
});

app.listen(PORT, () => {
  console.log('\n🚑 DisasterCare Server Started!');
  console.log(`📍 Server: http://localhost:${PORT}`);
  console.log('\n✅ Ready to handle emergencies!');
});
