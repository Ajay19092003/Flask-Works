import React, { useState, useEffect } from 'react'
import { Stethoscope, Send, Clock, Phone, MapPin, Activity, AlertTriangle, FileText } from 'lucide-react'
import axios from 'axios'
import toast from 'react-hot-toast'
import { useAuth } from '../context/AuthContext'
import { useSocket } from '../context/SocketContext'

const MedicalExpertDashboard = () => {
  const [activeCases, setActiveCases] = useState([])
  const [selectedCase, setSelectedCase] = useState(null)
  const [guidance, setGuidance] = useState('')
  const [recommendedHospital, setRecommendedHospital] = useState('')
  const [hospitals, setHospitals] = useState([])
  const [loading, setLoading] = useState(true)
  const { user } = useAuth()
  const { socket } = useSocket()

  useEffect(() => {
    fetchActiveCases()
    fetchHospitals()
  }, [])

  const fetchActiveCases = async () => {
    try {
      const response = await axios.get('/medical-expert/active-cases')
      setActiveCases(response.data)
    } catch (error) {
      console.error('Failed to fetch active cases:', error)
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

  const sendGuidance = async () => {
    if (!guidance.trim() || !selectedCase) {
      toast.error('Please enter guidance text')
      return
    }

    try {
      await axios.post('/medical-expert/provide-guidance', {
        emergencyId: selectedCase.id,
        guidance,
        recommendedHospital
      })
      
      toast.success('Medical guidance sent successfully!')
      setGuidance('')
      setRecommendedHospital('')
      
      // Refresh cases
      fetchActiveCases()
    } catch (error) {
      toast.error(error.response?.data?.error || 'Failed to send guidance')
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

  const getSeverityColor = (condition) => {
    if (!condition) return 'bg-gray-100 text-gray-800'
    
    if (condition.toLowerCase().includes('critical')) {
      return 'bg-red-100 text-red-800'
    } else if (condition.toLowerCase().includes('serious')) {
      return 'bg-orange-100 text-orange-800'
    } else if (condition.toLowerCase().includes('stable')) {
      return 'bg-green-100 text-green-800'
    }
    return 'bg-gray-100 text-gray-800'
  }

  const getQuickGuidanceTemplates = (condition) => {
    const templates = {
      'critical': [
        'Monitor vital signs continuously - check BP, pulse, breathing rate every 2 minutes',
        'Ensure airway is clear - use suction if necessary, position patient properly',
        'Be prepared for CPR - have defibrillator ready, check pulse regularly',
        'IV access essential - establish large bore IV, prepare for fluid resuscitation'
      ],
      'cardiac': [
        'Administer aspirin 324mg if not allergic and no bleeding risk',
        'Monitor ECG continuously - watch for arrhythmias, ST changes',
        'Prepare nitroglycerin - 0.4mg sublingual, may repeat every 5 minutes',
        'Keep patient calm and in semi-upright position'
      ],
      'trauma': [
        'Control bleeding - apply direct pressure, use tourniquets if needed',
        'Stabilize spine - maintain cervical spine immobilization',
        'Monitor for shock - watch BP, pulse, skin color and temperature',
        'Prepare for rapid transport - notify trauma center immediately'
      ],
      'pediatric': [
        'Use age-appropriate equipment - pediatric BP cuff, oxygen mask',
        'Monitor temperature closely - children lose heat rapidly',
        'Calculate drug doses by weight - verify all medications carefully',
        'Keep parent/caregiver informed and calm'
      ]
    }

    const conditionLower = condition?.toLowerCase() || ''
    
    if (conditionLower.includes('critical') || conditionLower.includes('life')) {
      return templates.critical
    } else if (conditionLower.includes('cardiac') || conditionLower.includes('chest')) {
      return templates.cardiac
    } else if (conditionLower.includes('trauma') || conditionLower.includes('accident')) {
      return templates.trauma
    } else if (conditionLower.includes('child') || conditionLower.includes('pediatric')) {
      return templates.pediatric
    }
    
    return templates.critical // Default to critical care
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
              Medical Expert Dashboard
            </h1>
            <p className="text-gray-600">
              Provide remote medical guidance during emergency transport
            </p>
          </div>

          {/* Stats Cards */}
          <div className="grid grid-4 gap-6 mb-8">
            <div className="stats-card">
              <div className="stats-number">{activeCases.length}</div>
              <div className="stats-label">Active Cases</div>
            </div>
            <div className="stats-card">
              <div className="stats-number">{activeCases.filter(c => c.health_condition?.includes('Critical')).length}</div>
              <div className="stats-label">Critical Cases</div>
            </div>
            <div className="stats-card">
              <div className="stats-number">24/7</div>
              <div className="stats-label">Availability</div>
            </div>
            <div className="stats-card">
              <div className="stats-number">{user.name.split(' ')[0]}</div>
              <div className="stats-label">On Duty</div>
            </div>
          </div>

          <div className="grid grid-1 lg:grid-3 gap-8">
            
            {/* Active Cases */}
            <div className="lg:col-span-2">
              <h2 className="text-xl font-semibold mb-4">Active Transport Cases</h2>
              
              {activeCases.length === 0 ? (
                <div className="card text-center py-8">
                  <Stethoscope className="w-12 h-12 text-gray-300 mx-auto mb-4" />
                  <p className="text-gray-500">No active transport cases</p>
                  <p className="text-sm text-gray-400 mt-2">You'll be notified when patients need medical guidance</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {activeCases.map((case_item) => (
                    <div 
                      key={case_item.id} 
                      className={`card cursor-pointer transition-all ${
                        selectedCase?.id === case_item.id ? 'ring-2 ring-blue-500 bg-blue-50' : 'hover:shadow-md'
                      }`}
                      onClick={() => setSelectedCase(case_item)}
                    >
                      <div className="flex items-start justify-between mb-4">
                        <div>
                          <h3 className="font-semibold text-lg">Emergency #{case_item.id.slice(-8)}</h3>
                          <p className="text-sm text-gray-600">
                            Patient: {case_item.victim_name}
                          </p>
                          <p className="text-sm text-gray-600">
                            Driver: {case_item.driver_name} • Vehicle: {case_item.vehicle_number}
                          </p>
                        </div>
                        <div className="flex flex-col gap-2">
                          <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(case_item.status)}`}>
                            {case_item.status.replace('_', ' ').toUpperCase()}
                          </span>
                          {case_item.health_condition && (
                            <span className={`px-3 py-1 rounded-full text-sm font-medium ${getSeverityColor(case_item.health_condition)}`}>
                              {case_item.health_condition.split(' - ')[0]}
                            </span>
                          )}
                        </div>
                      </div>

                      <div className="grid grid-2 gap-4 mb-4">
                        <div>
                          <p className="text-sm text-gray-500">Ambulance Type</p>
                          <p className="font-medium">{case_item.ambulance_type}</p>
                        </div>
                        <div>
                          <p className="text-sm text-gray-500">Condition</p>
                          <p className="font-medium">{case_item.health_condition || 'Not specified'}</p>
                        </div>
                      </div>

                      {case_item.emergency_details && (
                        <div className="mb-4">
                          <p className="text-sm text-gray-500">Emergency Details</p>
                          <p className="text-sm bg-gray-50 p-2 rounded">{case_item.emergency_details}</p>
                        </div>
                      )}

                      <div className="flex items-center gap-4 text-sm text-gray-600">
                        <div className="flex items-center gap-1">
                          <Clock size={14} />
                          <span>{new Date(case_item.created_at).toLocaleTimeString()}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <Activity size={14} />
                          <span>In Transit</span>
                        </div>
                      </div>

                      {selectedCase?.id === case_item.id && (
                        <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                          <p className="text-sm font-medium text-blue-800">
                            Click "Provide Guidance" to send medical instructions →
                          </p>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Medical Guidance Panel */}
            <div>
              <h2 className="text-xl font-semibold mb-4">Medical Guidance</h2>
              
              {selectedCase ? (
                <div className="space-y-4">
                  <div className="card">
                    <div className="card-header">
                      <h3 className="card-title">Patient Case #{selectedCase.id.slice(-8)}</h3>
                      <p className="text-sm text-gray-600">{selectedCase.victim_name}</p>
                    </div>

                    <div className="space-y-4">
                      <div>
                        <label className="form-label">Medical Guidance</label>
                        <textarea
                          className="form-input"
                          rows="4"
                          placeholder="Provide medical instructions for the driver..."
                          value={guidance}
                          onChange={(e) => setGuidance(e.target.value)}
                        />
                      </div>

                      <div>
                        <label className="form-label">Recommended Hospital (Optional)</label>
                        <select
                          className="form-select"
                          value={recommendedHospital}
                          onChange={(e) => setRecommendedHospital(e.target.value)}
                        >
                          <option value="">Select hospital if redirection needed</option>
                          {hospitals.map((hospital) => (
                            <option key={hospital.id} value={hospital.id}>
                              {hospital.name} - {hospital.specializations}
                            </option>
                          ))}
                        </select>
                      </div>

                      <button
                        onClick={sendGuidance}
                        disabled={!guidance.trim()}
                        className="btn btn-primary w-full"
                      >
                        <Send size={16} />
                        Send Guidance to Driver
                      </button>
                    </div>
                  </div>

                  {/* Quick Guidance Templates */}
                  <div className="card">
                    <h3 className="font-semibold mb-3">Quick Guidance Templates</h3>
                    <div className="space-y-2">
                      {getQuickGuidanceTemplates(selectedCase.health_condition).map((template, index) => (
                        <button
                          key={index}
                          onClick={() => setGuidance(template)}
                          className="w-full text-left p-2 text-sm bg-gray-50 hover:bg-gray-100 rounded border"
                        >
                          {template}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="card text-center py-8">
                  <FileText className="w-12 h-12 text-gray-300 mx-auto mb-4" />
                  <p className="text-gray-500">Select a case to provide guidance</p>
                  <p className="text-sm text-gray-400 mt-2">Choose from active transport cases on the left</p>
                </div>
              )}

              {/* Medical Protocols */}
              <div className="card mt-6">
                <h3 className="font-semibold mb-3">Emergency Protocols</h3>
                <div className="space-y-3 text-sm">
                  <div>
                    <h4 className="font-medium text-red-600">🚨 Critical Patients</h4>
                    <p className="text-gray-600">Continuous monitoring, airway management, IV access</p>
                  </div>
                  <div>
                    <h4 className="font-medium text-orange-600">⚠️ Cardiac Events</h4>
                    <p className="text-gray-600">ECG monitoring, aspirin, nitroglycerin protocols</p>
                  </div>
                  <div>
                    <h4 className="font-medium text-blue-600">🩺 Trauma Cases</h4>
                    <p className="text-gray-600">Bleeding control, spine stabilization, shock prevention</p>
                  </div>
                  <div>
                    <h4 className="font-medium text-purple-600">👶 Pediatric Care</h4>
                    <p className="text-gray-600">Age-appropriate equipment, weight-based dosing</p>
                  </div>
                </div>
              </div>

              {/* Expert Info */}
              <div className="card mt-6">
                <h3 className="font-semibold mb-3">Expert Information</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Name</span>
                    <span className="font-medium">{user.name}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Status</span>
                    <span className="font-medium text-green-600">Available</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Response Time</span>
                    <span className="font-medium">~2 minutes</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Active Cases</span>
                    <span className="font-medium">{activeCases.length}</span>
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

export default MedicalExpertDashboard