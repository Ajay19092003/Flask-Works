const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const sqlite3 = require('sqlite3').verbose();

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "http://localhost:5173",
    methods: ["GET", "POST"]
  }
});

const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Database initialization
const db = new sqlite3.Database('./disaster_management.db');

// Initialize database tables
db.serialize(() => {
  // Users table
  db.run(`CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    phone TEXT NOT NULL,
    password TEXT NOT NULL,
    role TEXT NOT NULL,
    age INTEGER,
    gender TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
  )`);

  // Hospitals table with sample data
  const hospitals = [
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
  ];

  db.run(`CREATE TABLE IF NOT EXISTS hospitals (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    address TEXT NOT NULL,
    lat REAL NOT NULL,
    lng REAL NOT NULL,
    contact_phone TEXT NOT NULL,
    specializations TEXT
  )`);

  hospitals.forEach(hospital => {
    db.run(`INSERT OR IGNORE INTO hospitals (id, name, address, lat, lng, contact_phone, specializations) 
            VALUES (?, ?, ?, ?, ?, ?, ?)`,
      [hospital.id, hospital.name, hospital.address, hospital.lat, hospital.lng, hospital.contact_phone, hospital.specializations]);
  });
});

// Socket handling
io.on('connection', (socket) => {
  console.log('User connected:', socket.id);
  
  socket.on('join_room', (data) => {
    socket.join(data.room);
    console.log(`User joined room: ${data.room}`);
  });

  socket.on('disconnect', () => {
    console.log('User disconnected:', socket.id);
  });
});

// Basic API routes
app.get('/', (req, res) => {
  res.json({ 
    message: '🚑 Disaster Management API is running!',
    version: '1.0.0',
    status: 'healthy'
  });
});

app.get('/api/hospitals', (req, res) => {
  db.all(`SELECT * FROM hospitals ORDER BY name`, (err, hospitals) => {
    if (err) {
      return res.status(500).json({ error: 'Database error' });
    }
    res.json(hospitals);
  });
});

app.get('/api/status', (req, res) => {
  res.json({
    server: 'running',
    database: 'connected',
    websocket: 'active',
    timestamp: new Date().toISOString()
  });
});

// Demo endpoint for testing
app.post('/api/emergency/create', (req, res) => {
  const { ambulanceType, lat, lng, healthCondition, emergencyDetails } = req.body;
  
  // Simple response for demo
  res.json({
    emergencyId: 'demo-' + Date.now(),
    message: 'Emergency request received! (Demo mode)',
    location: { lat, lng },
    ambulanceType,
    status: 'assigned'
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Something went wrong!' });
});

// Start server
server.listen(PORT, () => {
  console.log('\n🚑 DisasterCare Server Started!');
  console.log(`📍 Server: http://localhost:${PORT}`);
  console.log(`🔗 WebSocket: Ready for real-time communication`);
  console.log(`📊 Database: SQLite initialized`);
  console.log(`🏥 Sample hospitals loaded`);
  console.log('\n✅ Ready to handle emergencies!');
});