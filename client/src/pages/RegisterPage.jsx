import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { UserPlus, Mail, Lock, User, Phone, Calendar, Users } from 'lucide-react'
import toast from 'react-hot-toast'
import { useAuth } from '../context/AuthContext'

const RegisterPage = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    phone: '',
    password: '',
    confirmPassword: '',
    role: '',
    age: '',
    gender: '',
    additionalData: {}
  })
  const [loading, setLoading] = useState(false)
  const { register } = useAuth()
  const navigate = useNavigate()

  const handleChange = (e) => {
    const { name, value } = e.target
    
    if (name.startsWith('additional.')) {
      const fieldName = name.replace('additional.', '')
      setFormData({
        ...formData,
        additionalData: {
          ...formData.additionalData,
          [fieldName]: value
        }
      })
    } else {
      setFormData({
        ...formData,
        [name]: value
      })
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (formData.password !== formData.confirmPassword) {
      toast.error('Passwords do not match')
      return
    }
    
    setLoading(true)
    
    const result = await register(formData)
    
    if (result.success) {
      // Redirect based on user role
      switch (result.user.role) {
        case 'victim':
          navigate('/victim/dashboard')
          break
        case 'driver':
          navigate('/driver/dashboard')
          break
        case 'hospital':
          navigate('/hospital/dashboard')
          break
        case 'medical_expert':
          navigate('/expert/dashboard')
          break
        default:
          navigate('/')
      }
    }
    
    setLoading(false)
  }

  const renderRoleSpecificFields = () => {
    switch (formData.role) {
      case 'driver':
        return (
          <>
            <div className="form-group">
              <label className="form-label">Vehicle Number</label>
              <input
                type="text"
                name="additional.vehicleNumber"
                className="form-input"
                placeholder="e.g., AB-1234-CD"
                value={formData.additionalData.vehicleNumber || ''}
                onChange={handleChange}
                required
              />
            </div>
            <div className="form-group">
              <label className="form-label">Badge Number</label>
              <input
                type="text"
                name="additional.badgeNumber"
                className="form-input"
                placeholder="Driver badge number"
                value={formData.additionalData.badgeNumber || ''}
                onChange={handleChange}
                required
              />
            </div>
            <div className="form-group">
              <label className="form-label">License Number</label>
              <input
                type="text"
                name="additional.licenseNumber"
                className="form-input"
                placeholder="Driving license number"
                value={formData.additionalData.licenseNumber || ''}
                onChange={handleChange}
                required
              />
            </div>
            <div className="form-group">
              <label className="form-label">Ambulance Type</label>
              <select
                name="additional.ambulanceType"
                className="form-select"
                value={formData.additionalData.ambulanceType || ''}
                onChange={handleChange}
                required
              >
                <option value="">Select ambulance type</option>
                <option value="MICU">MICU (Mobile Intensive Care Unit)</option>
                <option value="Basic Unit">Basic Unit</option>
                <option value="Pediatric Unit">Pediatric Unit</option>
              </select>
            </div>
          </>
        )
      
      case 'medical_expert':
        return (
          <>
            <div className="form-group">
              <label className="form-label">Medical Specialization</label>
              <select
                name="additional.specialization"
                className="form-select"
                value={formData.additionalData.specialization || ''}
                onChange={handleChange}
                required
              >
                <option value="">Select specialization</option>
                <option value="Emergency Medicine">Emergency Medicine</option>
                <option value="Cardiology">Cardiology</option>
                <option value="Pediatrics">Pediatrics</option>
                <option value="Trauma Surgery">Trauma Surgery</option>
                <option value="General Medicine">General Medicine</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Medical License Number</label>
              <input
                type="text"
                name="additional.licenseNumber"
                className="form-input"
                placeholder="Medical license number"
                value={formData.additionalData.licenseNumber || ''}
                onChange={handleChange}
                required
              />
            </div>
          </>
        )
      
      default:
        return null
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-12">
      <div className="max-w-lg w-full">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Join DisasterCare</h1>
          <p className="text-gray-200">Create your account and start saving lives</p>
        </div>

        <div className="card">
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Basic Information */}
            <div className="grid grid-2 gap-4">
              <div className="form-group">
                <label className="form-label">
                  <User size={16} className="inline mr-2" />
                  Full Name
                </label>
                <input
                  type="text"
                  name="name"
                  className="form-input"
                  placeholder="Enter your full name"
                  value={formData.name}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label className="form-label">
                  <Phone size={16} className="inline mr-2" />
                  Phone Number
                </label>
                <input
                  type="tel"
                  name="phone"
                  className="form-input"
                  placeholder="+1 (555) 123-4567"
                  value={formData.phone}
                  onChange={handleChange}
                  required
                />
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">
                <Mail size={16} className="inline mr-2" />
                Email Address
              </label>
              <input
                type="email"
                name="email"
                className="form-input"
                placeholder="Enter your email"
                value={formData.email}
                onChange={handleChange}
                required
              />
            </div>

            <div className="grid grid-2 gap-4">
              <div className="form-group">
                <label className="form-label">
                  <Calendar size={16} className="inline mr-2" />
                  Age
                </label>
                <input
                  type="number"
                  name="age"
                  className="form-input"
                  placeholder="Age"
                  min="18"
                  max="100"
                  value={formData.age}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label className="form-label">
                  <Users size={16} className="inline mr-2" />
                  Gender
                </label>
                <select
                  name="gender"
                  className="form-select"
                  value={formData.gender}
                  onChange={handleChange}
                  required
                >
                  <option value="">Select gender</option>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                  <option value="other">Other</option>
                  <option value="prefer-not-to-say">Prefer not to say</option>
                </select>
              </div>
            </div>

            {/* Role Selection */}
            <div className="form-group">
              <label className="form-label">User Role</label>
              <select
                name="role"
                className="form-select"
                value={formData.role}
                onChange={handleChange}
                required
              >
                <option value="">Select your role</option>
                <option value="victim">Victim/Helper - Request emergency assistance</option>
                <option value="driver">Ambulance Driver - Respond to emergencies</option>
                <option value="hospital">Hospital Helpdesk - Manage incoming patients</option>
                <option value="medical_expert">Medical Expert - Provide remote guidance</option>
              </select>
            </div>

            {/* Role-specific fields */}
            {renderRoleSpecificFields()}

            {/* Password fields */}
            <div className="grid grid-2 gap-4">
              <div className="form-group">
                <label className="form-label">
                  <Lock size={16} className="inline mr-2" />
                  Password
                </label>
                <input
                  type="password"
                  name="password"
                  className="form-input"
                  placeholder="Create password"
                  value={formData.password}
                  onChange={handleChange}
                  required
                  minLength="6"
                />
              </div>

              <div className="form-group">
                <label className="form-label">Confirm Password</label>
                <input
                  type="password"
                  name="confirmPassword"
                  className="form-input"
                  placeholder="Confirm password"
                  value={formData.confirmPassword}
                  onChange={handleChange}
                  required
                  minLength="6"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="btn btn-primary w-full"
            >
              {loading ? (
                <div className="spinner w-5 h-5"></div>
              ) : (
                <>
                  <UserPlus size={20} />
                  Create Account
                </>
              )}
            </button>
          </form>

          <div className="mt-6 text-center">
            <p className="text-gray-600">
              Already have an account?{' '}
              <Link to="/login" className="text-blue-600 hover:text-blue-700 font-medium">
                Sign in here
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default RegisterPage