import React, { useState } from 'react'
import './App.css'

function App() {
  const [currentRole, setCurrentRole] = useState('landing')
  const [activeEmergency, setActiveEmergency] = useState(null)

  // Sample emergency data
  const sampleEmergency = {
    id: 'DEMO-' + Date.now(),
    location: 'Current Location (GPS)',
    ambulanceType: 'MICU',
    status: 'assigned',
    driver: 'Mike Driver',
    vehicle: 'AMB-001',
    eta: '8 minutes'
  }

  const handleEmergencyRequest = (type) => {
    setActiveEmergency({...sampleEmergency, ambulanceType: type})
  }

  const LandingPage = () => (
    <div className="landing-page">
      <div className="hero-section">
        <h1 className="hero-title">🚑 DisasterCare</h1>
        <p className="hero-subtitle">Emergency Management System</p>
        <p className="hero-description">
          Intelligent disaster management with boosted ambulance allocation, 
          real-time GPS tracking, and coordinated communication.
        </p>
        
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">🚨</div>
            <h3>Boosted Allocation</h3>
            <p>Smart algorithm prioritizes recently active ambulances for faster response</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">📍</div>
            <h3>Real-time GPS</h3>
            <p>Live location tracking and navigation for optimal emergency response</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🏥</div>
            <h3>3-Tier Hospital Alerts</h3>
            <p>Selection, proximity, and arrival notifications for seamless handover</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🩺</div>
            <h3>Remote Guidance</h3>
            <p>Medical expert consultation during patient transport</p>
          </div>
        </div>

        <div className="role-selector">
          <h2>Choose Your Role</h2>
          <div className="role-buttons">
            <button className="role-btn victim" onClick={() => setCurrentRole('victim')}>
              🆘 Victim/Helper
            </button>
            <button className="role-btn driver" onClick={() => setCurrentRole('driver')}>
              🚑 Ambulance Driver
            </button>
            <button className="role-btn hospital" onClick={() => setCurrentRole('hospital')}>
              🏥 Hospital Helpdesk
            </button>
            <button className="role-btn expert" onClick={() => setCurrentRole('expert')}>
              🩺 Medical Expert
            </button>
          </div>
        </div>
      </div>
    </div>
  )

  const VictimDashboard = () => (
    <div className="dashboard victim-dashboard">
      <div className="dashboard-header">
        <h1>🆘 Emergency Response Dashboard</h1>
        <button className="back-btn" onClick={() => setCurrentRole('landing')}>← Back to Home</button>
      </div>
      
      <div className="location-status">
        <div className="status-indicator">
          <span className="status-dot green"></span>
          <span>GPS Location: Ready (40.7128, -74.0060)</span>
        </div>
      </div>

      <div className="emergency-buttons">
        <button className="emergency-btn critical" onClick={() => handleEmergencyRequest('MICU')}>
          <div className="btn-icon">🚨</div>
          <div className="btn-text">
            <strong>CRITICAL</strong>
            <small>Life-threatening emergency</small>
          </div>
        </button>
        <button className="emergency-btn urgent" onClick={() => handleEmergencyRequest('Basic Unit')}>
          <div className="btn-icon">💔</div>
          <div className="btn-text">
            <strong>URGENT</strong>
            <small>Need medical attention</small>
          </div>
        </button>
        <button className="emergency-btn pediatric" onClick={() => handleEmergencyRequest('Pediatric Unit')}>
          <div className="btn-icon">👶</div>
          <div className="btn-text">
            <strong>PEDIATRIC</strong>
            <small>Child/infant emergency</small>
          </div>
        </button>
      </div>

      {activeEmergency && (
        <div className="emergency-status">
          <h3>🚑 Emergency Active</h3>
          <div className="status-card">
            <p><strong>Emergency ID:</strong> {activeEmergency.id}</p>
            <p><strong>Ambulance Type:</strong> {activeEmergency.ambulanceType}</p>
            <p><strong>Driver:</strong> {activeEmergency.driver}</p>
            <p><strong>Vehicle:</strong> {activeEmergency.vehicle}</p>
            <p><strong>ETA:</strong> {activeEmergency.eta}</p>
            <div className="status-indicator">
              <span className="status-dot green pulse"></span>
              <span>Driver en route - Real-time tracking active</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )

  const DriverDashboard = () => (
    <div className="dashboard driver-dashboard">
      <div className="dashboard-header">
        <h1>🚑 Driver Dashboard</h1>
        <button className="back-btn" onClick={() => setCurrentRole('landing')}>← Back to Home</button>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-number">2</div>
          <div className="stat-label">Active Requests</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">15</div>
          <div className="stat-label">Completed Today</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">GPS</div>
          <div className="stat-label">Location Status</div>
        </div>
      </div>

      <div className="emergency-requests">
        <h3>Emergency Assignments</h3>
        <div className="request-card">
          <div className="request-header">
            <h4>Emergency #12345</h4>
            <span className="status-badge assigned">ASSIGNED</span>
          </div>
          <div className="request-details">
            <p><strong>Patient:</strong> John Victim • +1-555-0001</p>
            <p><strong>Type:</strong> MICU - Critical condition</p>
            <p><strong>Location:</strong> 40.7128, -74.0060</p>
            <div className="request-actions">
              <button className="btn accept">✅ Accept & Navigate</button>
              <button className="btn navigate">🗺️ Maps</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )

  const HospitalDashboard = () => (
    <div className="dashboard hospital-dashboard">
      <div className="dashboard-header">
        <h1>🏥 Hospital Emergency Dashboard</h1>
        <button className="back-btn" onClick={() => setCurrentRole('landing')}>← Back to Home</button>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-number">3</div>
          <div className="stat-label">Incoming Ambulances</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">24</div>
          <div className="stat-label">Today's Emergencies</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">85%</div>
          <div className="stat-label">Bed Availability</div>
        </div>
      </div>

      <div className="alert-system">
        <h3>🚨 Hospital Alert System (3-Tier)</h3>
        <div className="alerts">
          <div className="alert-item selection">
            <span className="alert-icon">🚑</span>
            <div className="alert-content">
              <strong>Alert #1: Hospital Selection</strong>
              <p>Ambulance AMB-001 en route to your facility</p>
              <small>ETA: 12 minutes</small>
            </div>
          </div>
          <div className="alert-item proximity">
            <span className="alert-icon">📍</span>
            <div className="alert-content">
              <strong>Alert #2: 200m Proximity</strong>
              <p>Ambulance is approaching - Final preparation</p>
              <small>Arrival imminent</small>
            </div>
          </div>
          <div className="alert-item arrival">
            <span className="alert-icon">🏁</span>
            <div className="alert-content">
              <strong>Alert #3: Arrival</strong>
              <p>Ambulance has arrived at entrance</p>
              <small>Patient handover ready</small>
            </div>
          </div>
        </div>
      </div>
    </div>
  )

  const ExpertDashboard = () => (
    <div className="dashboard expert-dashboard">
      <div className="dashboard-header">
        <h1>🩺 Medical Expert Dashboard</h1>
        <button className="back-btn" onClick={() => setCurrentRole('landing')}>← Back to Home</button>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-number">5</div>
          <div className="stat-label">Active Cases</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">2</div>
          <div className="stat-label">Critical Cases</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">24/7</div>
          <div className="stat-label">Availability</div>
        </div>
      </div>

      <div className="guidance-panel">
        <h3>Remote Medical Guidance</h3>
        <div className="case-card">
          <div className="case-header">
            <h4>Emergency #12345 - Critical Patient</h4>
            <span className="condition-badge critical">CRITICAL</span>
          </div>
          <div className="case-details">
            <p><strong>Patient:</strong> John Victim</p>
            <p><strong>Condition:</strong> Cardiac arrest</p>
            <p><strong>Driver:</strong> Mike Driver (AMB-001)</p>
          </div>
          <div className="guidance-section">
            <textarea 
              placeholder="Provide medical instructions for the driver..."
              className="guidance-input"
            />
            <div className="quick-templates">
              <button className="template-btn">💊 Administer aspirin 324mg</button>
              <button className="template-btn">🩺 Monitor vital signs continuously</button>
              <button className="template-btn">🚨 Prepare for CPR if needed</button>
            </div>
            <button className="btn send-guidance">📤 Send Guidance to Driver</button>
          </div>
        </div>
      </div>
    </div>
  )

  const renderCurrentView = () => {
    switch (currentRole) {
      case 'victim': return <VictimDashboard />
      case 'driver': return <DriverDashboard />
      case 'hospital': return <HospitalDashboard />
      case 'expert': return <ExpertDashboard />
      default: return <LandingPage />
    }
  }

  return (
    <div className="app">
      {renderCurrentView()}
    </div>
  )
}

export default App