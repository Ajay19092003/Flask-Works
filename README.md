# 🚑 DisasterCare - Emergency Management System

A comprehensive disaster management web application with intelligent ambulance allocation, real-time GPS tracking, and coordinated communication between victims, drivers, hospitals, and medical experts.

## ✨ Features

### 🚨 **Boosted Allocation Algorithm**
- Prioritizes the last 10 ambulances in vicinity for faster response times
- Intelligent matching based on emergency type and proximity
- Real-time availability checking

### 👥 **Four User Roles**

#### 1. **Victim/Helper** 🆘
- Quick SOS buttons for emergency types (Critical, Urgent, Pediatric)
- Detailed emergency request forms
- Real-time ambulance tracking and status updates
- GPS-based location detection

#### 2. **Ambulance Driver** 🚑
- Emergency assignment notifications
- GPS navigation to victim location
- Patient status updates and hospital selection
- Real-time location sharing

#### 3. **Hospital Helpdesk** 🏥
- **3-Tier Alert System**:
  - Alert #1: Ambulance selection notification
  - Alert #2: 200m proximity warning
  - Alert #3: Arrival confirmation
- Incoming ambulance dashboard
- Patient preparation checklist

#### 4. **Medical Expert** 🩺
- Remote medical guidance during transport
- Quick guidance templates for different conditions
- Hospital recommendation capabilities
- Real-time case monitoring

### 🔧 **Technical Features**
- **Real-time Communication**: WebSocket-based live updates
- **GPS Tracking**: Continuous location monitoring
- **Responsive Design**: Works on desktop and mobile
- **Authentication**: JWT-based secure authentication
- **Database**: SQLite for easy setup (production-ready with PostgreSQL)

## 🚀 Quick Start

### Prerequisites
- Node.js (v16 or higher)
- npm or yarn

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd disaster-management-app
```

2. **Install dependencies**
```bash
npm run install-all
```

3. **Start the application**
```bash
npm run dev
```

This will start both the backend server (port 3001) and frontend client (port 5173).

### 🌐 Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:3001

## 👤 Demo Accounts

The application comes with pre-configured demo accounts for testing:

### Victim/Helper
- **Email**: victim@demo.com
- **Password**: password123

### Ambulance Driver  
- **Email**: driver@demo.com
- **Password**: password123

### Hospital Helpdesk
- **Email**: hospital@demo.com
- **Password**: password123

### Medical Expert
- **Email**: expert@demo.com  
- **Password**: password123

## 🏗️ Project Structure

```
disaster-management-app/
├── server/                 # Backend Node.js application
│   ├── index.js           # Main server file
│   ├── package.json       # Backend dependencies
│   └── disaster_management.db # SQLite database
├── client/                # Frontend React application
│   ├── src/
│   │   ├── components/    # Reusable UI components
│   │   ├── pages/         # Page components for each user role
│   │   ├── context/       # React contexts (Auth, Socket)
│   │   └── utils/         # Utility functions
│   ├── package.json       # Frontend dependencies
│   └── index.html         # HTML template
└── package.json           # Root package.json with scripts
```

## 🔄 Emergency Flow

### Complete System Workflow:

1. **🆘 Emergency Alert**
   - Victim raises SOS with ambulance type selection
   - GPS location automatically captured
   - Emergency details provided

2. **🤖 Smart Allocation**
   - Boosted algorithm analyzes last 10 active ambulances
   - Selects optimal ambulance based on type and proximity
   - Driver receives real-time assignment notification

3. **🚑 Response & Navigation**
   - Driver accepts request and navigates to victim
   - Real-time location updates shared with victim
   - Status updates throughout the journey

4. **🩺 Medical Guidance**
   - Medical expert provides remote consultation
   - Driver receives treatment instructions
   - Hospital recommendations if needed

5. **🏥 Hospital Coordination**
   - **Alert #1**: Hospital notified when selected as destination
   - **Alert #2**: 200m proximity warning for final preparation
   - **Alert #3**: Arrival confirmation for immediate handover

6. **✅ Completion**
   - Patient safely delivered to hospital
   - System updated and analytics recorded
   - Resources become available for next emergency

## 🛠️ API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login

### Emergency Management
- `POST /api/emergency/create` - Create emergency request
- `GET /api/emergency/:id/tracking` - Get emergency tracking

### Driver Operations
- `GET /api/driver/requests` - Get assigned requests
- `POST /api/driver/accept-request` - Accept emergency request
- `POST /api/driver/update-status` - Update emergency status

### Hospital Operations
- `GET /api/hospital/incoming` - Get incoming ambulances
- `GET /api/hospitals` - Get all hospitals

### Medical Expert
- `GET /api/medical-expert/active-cases` - Get active transport cases
- `POST /api/medical-expert/provide-guidance` - Send medical guidance

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the server directory:

```env
PORT=3001
JWT_SECRET=your_jwt_secret_key_here
NODE_ENV=development
```

### Database

The application uses SQLite by default for easy setup. For production, you can easily switch to PostgreSQL by updating the database configuration in `server/index.js`.

## 🚀 Deployment

### Production Build

1. **Build the frontend**
```bash
cd client && npm run build
```

2. **Start the production server**
```bash
cd server && npm start
```

### Docker Deployment (Optional)

Create a `Dockerfile` in the root directory for containerized deployment.

## 🧪 Testing

The application includes comprehensive testing scenarios:

1. **Emergency Request Flow**: Test complete SOS to hospital delivery
2. **Real-time Updates**: Verify WebSocket communications
3. **Alert System**: Test all three hospital alert types
4. **Medical Guidance**: Test expert consultation features

## 📱 Mobile Compatibility

The application is fully responsive and works on:
- ✅ Desktop browsers
- ✅ Mobile phones (iOS/Android)
- ✅ Tablets
- ✅ Progressive Web App (PWA) ready

## 🔒 Security Features

- **JWT Authentication**: Secure token-based authentication
- **Role-based Access**: Users can only access their designated features
- **Input Validation**: Server-side validation for all inputs
- **CORS Protection**: Configured for secure cross-origin requests

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For technical support or questions:
- Create an issue in the repository
- Check the documentation
- Review the demo account workflows

---

**Built with ❤️ for saving lives through technology**
