import React, { useState, useEffect } from 'react'
import { Truck, Clock, Users, Phone, MapPin, AlertTriangle, Activity } from 'lucide-react'
import axios from 'axios'
import toast from 'react-hot-toast'
import { useAuth } from '../context/AuthContext'
import { useSocket } from '../context/SocketContext'

const HospitalDashboard = () => {
  const [incomingAmbulances, setIncomingAmbulances] = useState([])
  const [stats, setStats] = useState({})
  const [loading, setLoading] = useState(true)
  const [alerts, setAlerts] = useState([])
  const { user } = useAuth()
  const { socket } = useSocket()

  useEffect(() => {
    fetchIncomingAmbulances()
    fetchStats()
  }, [])

  useEffect(() => {
    if (socket) {
      // Hospital Alert #1: Selection
      socket.on('hospital_alert_1', (data) => {
        const alert = {
          id: Date.now(),
          type: 'selection',
          message: data.message,
          emergencyId: data.emergencyId,
          patientCondition: data.patientCondition,
          timestamp: new Date()
        }
        setAlerts(prev => [alert, ...prev.slice(0, 9)]) // Keep last 10 alerts
        
        // Play sound and show notification
        toast.success('🚨 Ambulance en route to your hospital!', {
          duration: 10000,
          icon: '🏥',
        })
        
        fetchIncomingAmbulances() // Refresh data
      })

      // Hospital Alert #2: 200m proximity
      socket.on('hospital_alert_2', (data) => {
        const alert = {
          id: Date.now(),
          type: 'proximity',
          message: data.message,
          emergencyId: data.emergencyId,
          timestamp: new Date()
        }
        setAlerts(prev => [alert, ...prev.slice(0, 9)])
        
        toast.warning('⚠️ Ambulance is 200m away!', {
          duration: 8000,
          icon: '📍',
        })
      })

      // Hospital Alert #3: Arrival
      socket.on('hospital_alert_3', (data) => {
        const alert = {
          id: Date.now(),
          type: 'arrival',
          message: data.message,
          emergencyId: data.emergencyId,
          timestamp: new Date()
        }
        setAlerts(prev => [alert, ...prev.slice(0, 9)])
        
        toast.success('✅ Ambulance has arrived!', {
          duration: 6000,
          icon: '🏁',
        })
        
        fetchIncomingAmbulances() // Refresh data
      })

      return () => {
        socket.off('hospital_alert_1')
        socket.off('hospital_alert_2')
        socket.off('hospital_alert_3')
      }
    }
  }, [socket])

  const fetchIncomingAmbulances = async () => {
    try {
      const response = await axios.get('/hospital/incoming')
      setIncomingAmbulances(response.data)
    } catch (error) {
      console.error('Failed to fetch incoming ambulances:', error)
    } finally {
      setLoading(false)
    }
  }

  const fetchStats = async () => {
    try {
      const response = await axios.get('/dashboard/stats')
      setStats(response.data)
    } catch (error) {
      console.error('Failed to fetch stats:', error)
    }
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'patient_picked_up': return 'bg-yellow-100 text-yellow-800'
      case 'en_route_hospital': return 'bg-blue-100 text-blue-800'
      case 'completed': return 'bg-green-100 text-green-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  const getAlertIcon = (type) => {
    switch (type) {
      case 'selection': return '🚑'
      case 'proximity': return '📍'
      case 'arrival': return '🏁'
      default: return '🏥'
    }
  }

  const getAlertColor = (type) => {
    switch (type) {
      case 'selection': return 'bg-blue-50 border-blue-200'
      case 'proximity': return 'bg-orange-50 border-orange-200'
      case 'arrival': return 'bg-green-50 border-green-200'
      default: return 'bg-gray-50 border-gray-200'
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="spinner"></div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container">
        <div className="max-w-6xl mx-auto">
          
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-gray-800 mb-2">
              Hospital Emergency Dashboard
            </h1>
            <p className="text-gray-600">
              Monitor incoming ambulances and prepare for patient arrivals
            </p>
          </div>

          {/* Stats Cards */}
          <div className="grid grid-4 gap-6 mb-8">
            <div className="stats-card">
              <div className="stats-number">{incomingAmbulances.length}</div>
              <div className="stats-label">Incoming Ambulances</div>
            </div>
            <div className="stats-card">
              <div className="stats-number">{stats.todayEmergencies || 0}</div>
              <div className="stats-label">Today's Emergencies</div>
            </div>
            <div className="stats-card">
              <div className="stats-number">{stats.availableAmbulances || 0}</div>
              <div className="stats-label">Available Ambulances</div>
            </div>
            <div className="stats-card">
              <div className="stats-number">{stats.avgResponseTime || 0}min</div>
              <div className="stats-label">Avg Response Time</div>
            </div>
          </div>

          <div className="grid grid-1 lg:grid-3 gap-8">
            
            {/* Incoming Ambulances */}
            <div className="lg:col-span-2">
              <h2 className="text-xl font-semibold mb-4">Incoming Ambulances</h2>
              
              {incomingAmbulances.length === 0 ? (
                <div className="card text-center py-8">
                  <Truck className="w-12 h-12 text-gray-300 mx-auto mb-4" />
                  <p className="text-gray-500">No incoming ambulances</p>
                  <p className="text-sm text-gray-400 mt-2">You'll receive alerts when ambulances are en route</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {incomingAmbulances.map((ambulance) => (
                    <div key={ambulance.id} className="card">
                      <div className="flex items-start justify-between mb-4">
                        <div>
                          <h3 className="font-semibold text-lg">Emergency #{ambulance.id.slice(-8)}</h3>
                          <p className="text-sm text-gray-600">
                            Patient: {ambulance.victim_name}
                          </p>
                          <p className="text-sm text-gray-600">
                            Vehicle: {ambulance.vehicle_number}
                          </p>
                        </div>
                        <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(ambulance.status)}`}>
                          {ambulance.status.replace('_', ' ').toUpperCase()}
                        </span>
                      </div>

                      <div className="grid grid-2 gap-4 mb-4">
                        <div>
                          <p className="text-sm text-gray-500">Ambulance Type</p>
                          <p className="font-medium">{ambulance.ambulance_type}</p>
                        </div>
                        <div>
                          <p className="text-sm text-gray-500">Health Condition</p>
                          <p className="font-medium">{ambulance.health_condition || 'Not specified'}</p>
                        </div>
                      </div>

                      {ambulance.emergency_details && (
                        <div className="mb-4">
                          <p className="text-sm text-gray-500">Emergency Details</p>
                          <p className="text-sm">{ambulance.emergency_details}</p>
                        </div>
                      )}

                      <div className="grid grid-2 gap-4">
                        <div className="bg-gray-50 p-3 rounded-lg">
                          <h4 className="font-medium text-sm mb-2">Preparation Checklist</h4>
                          <div className="space-y-1 text-xs text-gray-600">
                            <div className="flex items-center gap-2">
                              <div className="w-4 h-4 bg-green-500 rounded-full flex items-center justify-center">
                                <span className="text-white text-xs">✓</span>
                              </div>
                              Emergency room prepared
                            </div>
                            <div className="flex items-center gap-2">
                              <div className="w-4 h-4 bg-green-500 rounded-full flex items-center justify-center">
                                <span className="text-white text-xs">✓</span>
                              </div>
                              Medical team on standby
                            </div>
                            <div className="flex items-center gap-2">
                              <div className="w-4 h-4 bg-yellow-500 rounded-full flex items-center justify-center">
                                <span className="text-white text-xs">!</span>
                              </div>
                              Equipment check in progress
                            </div>
                          </div>
                        </div>

                        <div className="bg-blue-50 p-3 rounded-lg">
                          <h4 className="font-medium text-sm mb-2">Estimated Arrival</h4>
                          <div className="flex items-center gap-2">
                            <Clock size={16} className="text-blue-600" />
                            <span className="text-sm font-medium text-blue-800">
                              {ambulance.status === 'patient_picked_up' ? '15-20 minutes' : 'Calculating...'}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Alert History */}
            <div>
              <h2 className="text-xl font-semibold mb-4">Recent Alerts</h2>
              
              <div className="space-y-3">
                {alerts.length === 0 ? (
                  <div className="card text-center py-6">
                    <AlertTriangle className="w-8 h-8 text-gray-300 mx-auto mb-2" />
                    <p className="text-sm text-gray-500">No recent alerts</p>
                  </div>
                ) : (
                  alerts.map((alert) => (
                    <div key={alert.id} className={`p-3 rounded-lg border ${getAlertColor(alert.type)}`}>
                      <div className="flex items-start gap-3">
                        <span className="text-lg">{getAlertIcon(alert.type)}</span>
                        <div className="flex-1">
                          <p className="font-medium text-sm">{alert.message}</p>
                          {alert.emergencyId && (
                            <p className="text-xs text-gray-600 mt-1">
                              Emergency #{alert.emergencyId.slice(-8)}
                            </p>
                          )}
                          {alert.patientCondition && (
                            <p className="text-xs text-gray-600">
                              Condition: {alert.patientCondition}
                            </p>
                          )}
                          <p className="text-xs text-gray-500 mt-1">
                            {alert.timestamp.toLocaleTimeString()}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>

              {/* Emergency Protocols */}
              <div className="card mt-6">
                <h3 className="font-semibold mb-3">Emergency Protocols</h3>
                <div className="space-y-3">
                  <div className="flex items-start gap-3">
                    <div className="w-6 h-6 bg-red-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                      <span className="text-red-600 text-xs font-bold">1</span>
                    </div>
                    <div>
                      <h4 className="font-medium text-sm">Alert Received</h4>
                      <p className="text-xs text-gray-600">Ambulance selection notification</p>
                    </div>
                  </div>
                  
                  <div className="flex items-start gap-3">
                    <div className="w-6 h-6 bg-orange-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                      <span className="text-orange-600 text-xs font-bold">2</span>
                    </div>
                    <div>
                      <h4 className="font-medium text-sm">200m Proximity</h4>
                      <p className="text-xs text-gray-600">Final preparation phase</p>
                    </div>
                  </div>
                  
                  <div className="flex items-start gap-3">
                    <div className="w-6 h-6 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                      <span className="text-green-600 text-xs font-bold">3</span>
                    </div>
                    <div>
                      <h4 className="font-medium text-sm">Arrival</h4>
                      <p className="text-xs text-gray-600">Immediate patient handover</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Hospital Stats */}
              <div className="card mt-6">
                <h3 className="font-semibold mb-3">Today's Activity</h3>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Patients Received</span>
                    <span className="font-medium">{stats.todayEmergencies || 0}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Average Stay</span>
                    <span className="font-medium">2.5 hours</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Bed Availability</span>
                    <span className="font-medium text-green-600">85%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Staff on Duty</span>
                    <span className="font-medium">24/7</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default HospitalDashboard