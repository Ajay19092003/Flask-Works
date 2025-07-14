import React, { useState, useEffect } from 'react'
import { Navigation, Phone, MapPin, Clock, CheckCircle, AlertTriangle, Users } from 'lucide-react'
import axios from 'axios'
import toast from 'react-hot-toast'
import { useAuth } from '../context/AuthContext'
import { useSocket } from '../context/SocketContext'

const DriverDashboard = () => {
  const [emergencyRequests, setEmergencyRequests] = useState([])
  const [hospitals, setHospitals] = useState([])
  const [loading, setLoading] = useState(true)
  const [currentLocation, setCurrentLocation] = useState(null)
  const { user } = useAuth()
  const { socket, updateLocation } = useSocket()

  useEffect(() => {
    fetchEmergencyRequests()
    fetchHospitals()
    getCurrentLocation()
  }, [])

  useEffect(() => {
    if (socket) {
      socket.on('new_emergency_assignment', (data) => {
        toast.success('🚨 New Emergency Assignment!', {
          duration: 8000,
          icon: '🚑',
        })
        fetchEmergencyRequests() // Refresh the list
      })

      socket.on('medical_guidance', (data) => {
        toast.success(`📋 Medical Guidance: ${data.guidance}`, {
          duration: 10000,
          icon: '🩺',
        })
      })

      return () => {
        socket.off('new_emergency_assignment')
        socket.off('medical_guidance')
      }
    }
  }, [socket])

  const getCurrentLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const location = {
            lat: position.coords.latitude,
            lng: position.coords.longitude
          }
          setCurrentLocation(location)
          
          // Update location in real-time if there's an active emergency
          if (emergencyRequests.length > 0) {
            updateLocation(location.lat, location.lng, emergencyRequests[0].id, user.id)
          }
        },
        (error) => {
          console.error('Error getting location:', error)
        }
      )
    }
  }

  const fetchEmergencyRequests = async () => {
    try {
      const response = await axios.get('/driver/requests')
      setEmergencyRequests(response.data)
    } catch (error) {
      console.error('Failed to fetch emergency requests:', error)
    } finally {
      setLoading(false)
    }
  }

  const fetchHospitals = async () => {
    try {
      const response = await axios.get('/hospitals')
      setHospitals(response.data)
    } catch (error) {
      console.error('Failed to fetch hospitals:', error)
    }
  }

  const acceptRequest = async (emergencyId) => {
    try {
      await axios.post('/driver/accept-request', { emergencyId })
      toast.success('Emergency request accepted!')
      fetchEmergencyRequests()
    } catch (error) {
      toast.error(error.response?.data?.error || 'Failed to accept request')
    }
  }

  const updateStatus = async (emergencyId, status, patientCondition, destinationHospitalId) => {
    try {
      await axios.post('/driver/update-status', {
        emergencyId,
        status,
        patientCondition,
        destinationHospitalId
      })
      toast.success('Status updated successfully!')
      fetchEmergencyRequests()
    } catch (error) {
      toast.error(error.response?.data?.error || 'Failed to update status')
    }
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'assigned': return 'bg-blue-100 text-blue-800'
      case 'en_route': return 'bg-yellow-100 text-yellow-800'
      case 'patient_picked_up': return 'bg-green-100 text-green-800'
      case 'completed': return 'bg-gray-100 text-gray-800'
      default: return 'bg-gray-100 text-gray-800'
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
              Driver Dashboard
            </h1>
            <p className="text-gray-600">
              Manage emergency responses and patient transport
            </p>
          </div>

          {/* Status Cards */}
          <div className="grid grid-4 gap-6 mb-8">
            <div className="stats-card">
              <div className="stats-number">{emergencyRequests.length}</div>
              <div className="stats-label">Active Requests</div>
            </div>
            <div className="stats-card">
              <div className="stats-number">{emergencyRequests.filter(r => r.status === 'completed').length}</div>
              <div className="stats-label">Completed Today</div>
            </div>
            <div className="stats-card">
              <div className="stats-number">{hospitals.length}</div>
              <div className="stats-label">Available Hospitals</div>
            </div>
            <div className="stats-card">
              <div className="stats-number">{currentLocation ? 'GPS' : 'No GPS'}</div>
              <div className="stats-label">Location Status</div>
            </div>
          </div>

          {/* Active Emergency Requests */}
          <div className="grid grid-1 lg:grid-2 gap-8">
            <div>
              <h2 className="text-xl font-semibold mb-4">Emergency Requests</h2>
              
              {emergencyRequests.length === 0 ? (
                <div className="card text-center py-8">
                  <AlertTriangle className="w-12 h-12 text-gray-300 mx-auto mb-4" />
                  <p className="text-gray-500">No active emergency requests</p>
                  <p className="text-sm text-gray-400 mt-2">You'll be notified when new emergencies are assigned</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {emergencyRequests.map((request) => (
                    <div key={request.id} className="card emergency-card">
                      <div className="flex items-start justify-between mb-4">
                        <div>
                          <h3 className="font-semibold text-lg">Emergency #{request.id.slice(-8)}</h3>
                          <p className="text-sm text-gray-600">
                            {request.victim_name} • {request.victim_phone}
                          </p>
                        </div>
                        <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(request.status)}`}>
                          {request.status.replace('_', ' ').toUpperCase()}
                        </span>
                      </div>

                      <div className="grid grid-2 gap-4 mb-4">
                        <div>
                          <p className="text-sm text-gray-500">Ambulance Type</p>
                          <p className="font-medium">{request.ambulance_type}</p>
                        </div>
                        <div>
                          <p className="text-sm text-gray-500">Health Condition</p>
                          <p className="font-medium">{request.health_condition || 'Not specified'}</p>
                        </div>
                      </div>

                      {request.emergency_details && (
                        <div className="mb-4">
                          <p className="text-sm text-gray-500">Emergency Details</p>
                          <p className="text-sm">{request.emergency_details}</p>
                        </div>
                      )}

                      <div className="flex items-center gap-2 mb-4">
                        <MapPin size={16} className="text-gray-400" />
                        <span className="text-sm text-gray-600">
                          {request.victim_lat.toFixed(4)}, {request.victim_lng.toFixed(4)}
                        </span>
                      </div>

                      {/* Action buttons based on status */}
                      <div className="flex gap-2">
                        {request.status === 'assigned' && (
                          <>
                            <button
                              onClick={() => acceptRequest(request.id)}
                              className="btn btn-success flex-1"
                            >
                              <CheckCircle size={16} />
                              Accept & Navigate
                            </button>
                            <button
                              onClick={() => window.open(`https://maps.google.com/?q=${request.victim_lat},${request.victim_lng}`, '_blank')}
                              className="btn btn-secondary"
                            >
                              <Navigation size={16} />
                            </button>
                          </>
                        )}

                        {request.status === 'en_route' && (
                          <button
                            onClick={() => updateStatus(request.id, 'patient_picked_up', 'Patient loaded', null)}
                            className="btn btn-success flex-1"
                          >
                            <Users size={16} />
                            Patient Picked Up
                          </button>
                        )}

                        {request.status === 'patient_picked_up' && (
                          <div className="space-y-2 w-full">
                            <div className="grid grid-2 gap-2">
                              <select
                                className="form-select"
                                onChange={(e) => {
                                  if (e.target.value) {
                                    updateStatus(request.id, 'en_route_hospital', 'En route to hospital', e.target.value)
                                  }
                                }}
                              >
                                <option value="">Select Hospital</option>
                                {hospitals.map((hospital) => (
                                  <option key={hospital.id} value={hospital.id}>
                                    {hospital.name}
                                  </option>
                                ))}
                              </select>
                              <button
                                onClick={() => updateStatus(request.id, 'completed', 'Patient delivered', request.destination_hospital_id)}
                                className="btn btn-success"
                                disabled={!request.destination_hospital_id}
                              >
                                Complete
                              </button>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Hospital Information */}
            <div>
              <h2 className="text-xl font-semibold mb-4">Nearby Hospitals</h2>
              
              <div className="space-y-4">
                {hospitals.slice(0, 5).map((hospital) => (
                  <div key={hospital.id} className="card">
                    <div className="flex items-start justify-between">
                      <div>
                        <h3 className="font-semibold">{hospital.name}</h3>
                        <p className="text-sm text-gray-600 mb-2">{hospital.address}</p>
                        <div className="flex items-center gap-2">
                          <Phone size={14} className="text-gray-400" />
                          <span className="text-sm text-gray-600">{hospital.contact_phone}</span>
                        </div>
                        {hospital.specializations && (
                          <div className="mt-2">
                            <div className="flex flex-wrap gap-1">
                              {hospital.specializations.split(',').map((spec, index) => (
                                <span key={index} className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded">
                                  {spec.trim()}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                      <button
                        onClick={() => window.open(`https://maps.google.com/?q=${hospital.lat},${hospital.lng}`, '_blank')}
                        className="btn btn-secondary btn-sm"
                      >
                        <Navigation size={14} />
                      </button>
                    </div>
                  </div>
                ))}
              </div>

              {/* Driver Tips */}
              <div className="card mt-6">
                <h3 className="font-semibold mb-3">Driver Guidelines</h3>
                <ul className="text-sm text-gray-600 space-y-2">
                  <li>• Always use sirens and emergency lights when responding</li>
                  <li>• Follow traffic safety protocols even in emergency situations</li>
                  <li>• Communicate with medical experts for critical patients</li>
                  <li>• Update patient status regularly during transport</li>
                  <li>• Ensure proper handover documentation at hospitals</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default DriverDashboard