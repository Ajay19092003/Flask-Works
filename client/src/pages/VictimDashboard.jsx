import React, { useState, useEffect } from 'react'
import { AlertTriangle, MapPin, Phone, Clock, Heart, Activity } from 'lucide-react'
import axios from 'axios'
import toast from 'react-hot-toast'
import { useAuth } from '../context/AuthContext'
import { useSocket } from '../context/SocketContext'

const VictimDashboard = () => {
  const [location, setLocation] = useState(null)
  const [emergencyForm, setEmergencyForm] = useState({
    ambulanceType: '',
    healthCondition: '',
    emergencyDetails: ''
  })
  const [activeEmergency, setActiveEmergency] = useState(null)
  const [loading, setLoading] = useState(false)
  const { user } = useAuth()
  const { socket, connected } = useSocket()

  // Get user location on component mount
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setLocation({
            lat: position.coords.latitude,
            lng: position.coords.longitude
          })
        },
        (error) => {
          console.error('Error getting location:', error)
          toast.error('Unable to get your location. Please enable location services.')
        }
      )
    } else {
      toast.error('Geolocation is not supported by this browser.')
    }
  }, [])

  // Listen for socket events
  useEffect(() => {
    if (socket) {
      socket.on('driver_accepted', (data) => {
        toast.success('🚑 Driver accepted your request and is on the way!')
        // Refresh emergency status
        // You could fetch updated emergency data here
      })

      socket.on('driver_location_update', (data) => {
        // Update driver location on map
        console.log('Driver location update:', data)
      })

      return () => {
        socket.off('driver_accepted')
        socket.off('driver_location_update')
      }
    }
  }, [socket])

  const handleFormChange = (e) => {
    setEmergencyForm({
      ...emergencyForm,
      [e.target.name]: e.target.value
    })
  }

  const handleSOSSubmit = async (e) => {
    e.preventDefault()
    
    if (!location) {
      toast.error('Location not available. Please enable location services.')
      return
    }

    if (!emergencyForm.ambulanceType) {
      toast.error('Please select an ambulance type.')
      return
    }

    setLoading(true)

    try {
      const response = await axios.post('/emergency/create', {
        ...emergencyForm,
        lat: location.lat,
        lng: location.lng
      })

      setActiveEmergency(response.data)
      toast.success('🚨 Emergency request sent! Ambulance is being dispatched.')
      
      // Join emergency room for real-time updates
      if (socket) {
        socket.emit('join_room', { room: `emergency_${response.data.emergencyId}` })
      }

    } catch (error) {
      toast.error(error.response?.data?.error || 'Failed to send emergency request')
    } finally {
      setLoading(false)
    }
  }

  const quickSOS = async (type, condition) => {
    if (!location) {
      toast.error('Location not available. Please enable location services.')
      return
    }

    setLoading(true)

    try {
      const response = await axios.post('/emergency/create', {
        ambulanceType: type,
        healthCondition: condition,
        emergencyDetails: `Quick SOS - ${condition}`,
        lat: location.lat,
        lng: location.lng
      })

      setActiveEmergency(response.data)
      toast.success('🚨 Emergency request sent! Ambulance is being dispatched.')

    } catch (error) {
      toast.error(error.response?.data?.error || 'Failed to send emergency request')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container">
        <div className="max-w-4xl mx-auto">
          
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-gray-800 mb-2">
              Emergency Response Dashboard
            </h1>
            <p className="text-gray-600">
              Request immediate medical assistance in case of emergency
            </p>
          </div>

          {/* Location Status */}
          <div className="card mb-6">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
                <MapPin className={`w-6 h-6 ${location ? 'text-green-600' : 'text-red-600'}`} />
              </div>
              <div>
                <h3 className="font-semibold">Location Status</h3>
                <p className="text-gray-600">
                  {location 
                    ? `Location detected: ${location.lat.toFixed(4)}, ${location.lng.toFixed(4)}`
                    : 'Location not available'
                  }
                </p>
              </div>
              <div className="ml-auto">
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                  location ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                }`}>
                  {location ? 'Ready' : 'Not Ready'}
                </span>
              </div>
            </div>
          </div>

          {/* Quick SOS Buttons */}
          <div className="grid grid-3 gap-4 mb-8">
            <button
              onClick={() => quickSOS('MICU', 'Critical - Life threatening')}
              disabled={!location || loading}
              className="bg-red-500 hover:bg-red-600 text-white p-6 rounded-lg font-semibold text-center transition-colors disabled:opacity-50"
            >
              <AlertTriangle className="w-8 h-8 mx-auto mb-2" />
              <div>CRITICAL</div>
              <div className="text-sm opacity-90">Life-threatening emergency</div>
            </button>

            <button
              onClick={() => quickSOS('Basic Unit', 'Stable - Need transport')}
              disabled={!location || loading}
              className="bg-orange-500 hover:bg-orange-600 text-white p-6 rounded-lg font-semibold text-center transition-colors disabled:opacity-50"
            >
              <Heart className="w-8 h-8 mx-auto mb-2" />
              <div>URGENT</div>
              <div className="text-sm opacity-90">Need medical attention</div>
            </button>

            <button
              onClick={() => quickSOS('Pediatric Unit', 'Child emergency')}
              disabled={!location || loading}
              className="bg-purple-500 hover:bg-purple-600 text-white p-6 rounded-lg font-semibold text-center transition-colors disabled:opacity-50"
            >
              <Activity className="w-8 h-8 mx-auto mb-2" />
              <div>PEDIATRIC</div>
              <div className="text-sm opacity-90">Child/infant emergency</div>
            </button>
          </div>

          <div className="grid grid-2 gap-8">
            {/* Emergency Request Form */}
            <div className="card">
              <div className="card-header">
                <h2 className="card-title">Detailed Emergency Request</h2>
                <p className="text-gray-600">Provide specific details for better assistance</p>
              </div>

              <form onSubmit={handleSOSSubmit} className="space-y-4">
                <div className="form-group">
                  <label className="form-label">Ambulance Type Required</label>
                  <select
                    name="ambulanceType"
                    className="form-select"
                    value={emergencyForm.ambulanceType}
                    onChange={handleFormChange}
                    required
                  >
                    <option value="">Select ambulance type</option>
                    <option value="MICU">MICU - Mobile Intensive Care Unit</option>
                    <option value="Basic Unit">Basic Unit - Standard Transport</option>
                    <option value="Pediatric Unit">Pediatric Unit - Child Care</option>
                  </select>
                </div>

                <div className="form-group">
                  <label className="form-label">Health Condition</label>
                  <select
                    name="healthCondition"
                    className="form-select"
                    value={emergencyForm.healthCondition}
                    onChange={handleFormChange}
                  >
                    <option value="">Select condition severity</option>
                    <option value="Critical - Life threatening">Critical - Life threatening</option>
                    <option value="Serious - Urgent care needed">Serious - Urgent care needed</option>
                    <option value="Stable - Need transport">Stable - Need transport</option>
                    <option value="Minor - Non-urgent">Minor - Non-urgent</option>
                  </select>
                </div>

                <div className="form-group">
                  <label className="form-label">Emergency Details</label>
                  <textarea
                    name="emergencyDetails"
                    className="form-input"
                    rows="3"
                    placeholder="Describe the emergency situation, injuries, symptoms, etc."
                    value={emergencyForm.emergencyDetails}
                    onChange={handleFormChange}
                  ></textarea>
                </div>

                <button
                  type="submit"
                  disabled={!location || loading}
                  className="btn btn-primary w-full"
                >
                  {loading ? (
                    <div className="spinner w-5 h-5"></div>
                  ) : (
                    <>
                      <AlertTriangle size={20} />
                      Send Emergency Request
                    </>
                  )}
                </button>
              </form>
            </div>

            {/* Active Emergency Status */}
            <div className="card">
              <div className="card-header">
                <h2 className="card-title">Emergency Status</h2>
              </div>

              {activeEmergency ? (
                <div className="space-y-4">
                  <div className="alert alert-info">
                    <AlertTriangle size={20} />
                    <div>
                      <strong>Emergency Active</strong>
                      <p>Emergency ID: {activeEmergency.emergencyId}</p>
                    </div>
                  </div>

                  {activeEmergency.assignedDriver && (
                    <div className="space-y-3">
                      <h4 className="font-semibold">Assigned Ambulance</h4>
                      <div className="bg-gray-50 p-4 rounded-lg">
                        <div className="flex items-center gap-3">
                          <Phone size={16} />
                          <div>
                            <p className="font-medium">{activeEmergency.assignedDriver.name}</p>
                            <p className="text-sm text-gray-600">{activeEmergency.assignedDriver.phone}</p>
                            <p className="text-sm text-gray-600">Vehicle: {activeEmergency.assignedDriver.vehicleNumber}</p>
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-2 text-sm text-gray-600">
                        <Clock size={16} />
                        <span>Estimated distance: {Math.round(activeEmergency.assignedDriver.estimatedDistance)}m</span>
                      </div>
                    </div>
                  )}

                  {/* Real-time tracking would go here */}
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <p className="text-sm text-blue-800">
                      📍 Real-time tracking is active. You will receive updates as the ambulance approaches.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="text-center py-8">
                  <AlertTriangle className="w-12 h-12 text-gray-300 mx-auto mb-4" />
                  <p className="text-gray-500">No active emergency requests</p>
                  <p className="text-sm text-gray-400 mt-2">
                    Use the quick SOS buttons or detailed form to request help
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Emergency Tips */}
          <div className="card mt-8">
            <div className="card-header">
              <h3 className="card-title">Emergency Tips</h3>
            </div>
            <div className="grid grid-2 gap-6">
              <div>
                <h4 className="font-semibold mb-2">Before Emergency Services Arrive:</h4>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• Stay calm and keep the victim comfortable</li>
                  <li>• Do not move seriously injured persons</li>
                  <li>• Apply direct pressure to bleeding wounds</li>
                  <li>• Monitor breathing and consciousness</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold mb-2">When Ambulance Arrives:</h4>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• Have identification and medical information ready</li>
                  <li>• Provide clear directions to your location</li>
                  <li>• Inform them of any allergies or medications</li>
                  <li>• Stay out of the way but be available for questions</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default VictimDashboard