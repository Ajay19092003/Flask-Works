import React from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, Heart, Truck, Building2, Stethoscope, MapPin, Clock, Shield } from 'lucide-react'

const LandingPage = () => {
  const features = [
    {
      icon: MapPin,
      title: 'GPS Tracking',
      description: 'Real-time location tracking for ambulances and victims with optimized routing'
    },
    {
      icon: Clock,
      title: 'Boosted Allocation',
      description: 'Smart algorithm prioritizes recently active ambulances for faster response times'
    },
    {
      icon: Shield,
      title: '3-Tier Hospital Alerts',
      description: 'Automated alert system for hospitals at selection, proximity, and arrival'
    }
  ]

  const roles = [
    {
      icon: Heart,
      title: 'Victim/Helper',
      description: 'Quick SOS alerts with automatic ambulance dispatch based on emergency type',
      color: 'bg-red-100 text-red-600'
    },
    {
      icon: Truck,
      title: 'Ambulance Driver',
      description: 'Receive emergency assignments with GPS navigation and patient status updates',
      color: 'bg-blue-100 text-blue-600'
    },
    {
      icon: Building2,
      title: 'Hospital Helpdesk',
      description: 'Real-time alerts for incoming ambulances with patient condition details',
      color: 'bg-green-100 text-green-600'
    },
    {
      icon: Stethoscope,
      title: 'Medical Expert',
      description: 'Provide remote medical guidance during patient transport',
      color: 'bg-purple-100 text-purple-600'
    }
  ]

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="py-20 px-4">
        <div className="container">
          <div className="text-center max-w-4xl mx-auto">
            <h1 className="text-5xl font-bold text-white mb-6 leading-tight">
              🚑 Emergency Response
              <br />
              <span className="text-yellow-300">Reimagined</span>
            </h1>
            <p className="text-xl text-gray-200 mb-8 leading-relaxed">
              Intelligent disaster management system with boosted ambulance allocation, 
              real-time GPS tracking, and coordinated communication between all stakeholders.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link to="/register" className="btn btn-primary btn-lg">
                Get Started
                <ArrowRight size={20} />
              </Link>
              <Link to="/login" className="btn btn-secondary btn-lg">
                Sign In
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 px-4 bg-white">
        <div className="container">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-800 mb-4">
              Advanced Emergency Management
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Our system combines cutting-edge technology with proven emergency response protocols
            </p>
          </div>
          
          <div className="grid grid-3 gap-8">
            {features.map((feature, index) => (
              <div key={index} className="card text-center">
                <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <feature.icon size={32} className="text-blue-600" />
                </div>
                <h3 className="text-xl font-semibold mb-3">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Roles Section */}
      <section className="py-16 px-4 bg-gray-50">
        <div className="container">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-800 mb-4">
              Four User Roles, One Mission
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Each role is designed for specific responsibilities in the emergency response chain
            </p>
          </div>
          
          <div className="grid grid-2 gap-8">
            {roles.map((role, index) => (
              <div key={index} className="card hover:shadow-lg transition-shadow">
                <div className="flex items-start gap-4">
                  <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${role.color}`}>
                    <role.icon size={24} />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold mb-2">{role.title}</h3>
                    <p className="text-gray-600">{role.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-16 px-4 bg-white">
        <div className="container">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-800 mb-4">
              How It Works
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto">
              From emergency alert to hospital arrival - every step is optimized for speed and efficiency
            </p>
          </div>
          
          <div className="max-w-4xl mx-auto">
            <div className="grid grid-1 md:grid-2 gap-8">
              <div className="space-y-6">
                <div className="flex items-start gap-4">
                  <div className="w-8 h-8 bg-red-500 text-white rounded-full flex items-center justify-center font-bold">
                    1
                  </div>
                  <div>
                    <h4 className="font-semibold">Emergency Alert</h4>
                    <p className="text-gray-600">Victim raises SOS with location and ambulance type</p>
                  </div>
                </div>
                
                <div className="flex items-start gap-4">
                  <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center font-bold">
                    2
                  </div>
                  <div>
                    <h4 className="font-semibold">Smart Allocation</h4>
                    <p className="text-gray-600">Boosted algorithm selects optimal ambulance</p>
                  </div>
                </div>
                
                <div className="flex items-start gap-4">
                  <div className="w-8 h-8 bg-green-500 text-white rounded-full flex items-center justify-center font-bold">
                    3
                  </div>
                  <div>
                    <h4 className="font-semibold">Real-time Tracking</h4>
                    <p className="text-gray-600">GPS navigation and status updates</p>
                  </div>
                </div>
              </div>
              
              <div className="space-y-6">
                <div className="flex items-start gap-4">
                  <div className="w-8 h-8 bg-purple-500 text-white rounded-full flex items-center justify-center font-bold">
                    4
                  </div>
                  <div>
                    <h4 className="font-semibold">Medical Guidance</h4>
                    <p className="text-gray-600">Remote expert consultation during transport</p>
                  </div>
                </div>
                
                <div className="flex items-start gap-4">
                  <div className="w-8 h-8 bg-yellow-500 text-white rounded-full flex items-center justify-center font-bold">
                    5
                  </div>
                  <div>
                    <h4 className="font-semibold">Hospital Alerts</h4>
                    <p className="text-gray-600">3-tier notification system for preparation</p>
                  </div>
                </div>
                
                <div className="flex items-start gap-4">
                  <div className="w-8 h-8 bg-gray-500 text-white rounded-full flex items-center justify-center font-bold">
                    6
                  </div>
                  <div>
                    <h4 className="font-semibold">Patient Handover</h4>
                    <p className="text-gray-600">Seamless transfer to medical facility</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 px-4 bg-gradient-to-r from-red-500 to-orange-500">
        <div className="container">
          <div className="text-center text-white">
            <h2 className="text-3xl font-bold mb-4">
              Ready to Save Lives?
            </h2>
            <p className="text-xl mb-8 opacity-90">
              Join our emergency response network and make a difference
            </p>
            <Link to="/register" className="btn bg-white text-red-500 hover:bg-gray-100">
              Start Today
              <ArrowRight size={20} />
            </Link>
          </div>
        </div>
      </section>
    </div>
  )
}

export default LandingPage