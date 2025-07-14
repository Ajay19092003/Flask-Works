const sqlite3 = require('sqlite3').verbose();
const bcrypt = require('bcryptjs');
const { v4: uuidv4 } = require('uuid');

const db = new sqlite3.Database('./disaster_management.db');

const seedDemoUsers = async () => {
  console.log('🌱 Seeding demo users...');

  // Demo users data
  const demoUsers = [
    {
      id: uuidv4(),
      name: 'John Victim',
      email: 'victim@demo.com',
      phone: '+1-555-0001',
      password: await bcrypt.hash('password123', 10),
      role: 'victim',
      age: 35,
      gender: 'male'
    },
    {
      id: uuidv4(),
      name: 'Mike Driver',
      email: 'driver@demo.com',
      phone: '+1-555-0002',
      password: await bcrypt.hash('password123', 10),
      role: 'driver',
      age: 42,
      gender: 'male'
    },
    {
      id: uuidv4(),
      name: 'Sarah Hospital',
      email: 'hospital@demo.com',
      phone: '+1-555-0003',
      password: await bcrypt.hash('password123', 10),
      role: 'hospital',
      age: 38,
      gender: 'female'
    },
    {
      id: uuidv4(),
      name: 'Dr. Emily Expert',
      email: 'expert@demo.com',
      phone: '+1-555-0004',
      password: await bcrypt.hash('password123', 10),
      role: 'medical_expert',
      age: 45,
      gender: 'female'
    }
  ];

  // Insert demo users
  for (const user of demoUsers) {
    await new Promise((resolve, reject) => {
      db.run(
        `INSERT OR REPLACE INTO users (id, name, email, phone, password, role, age, gender) 
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
        [user.id, user.name, user.email, user.phone, user.password, user.role, user.age, user.gender],
        function(err) {
          if (err) {
            console.error(`Error inserting user ${user.name}:`, err);
            reject(err);
          } else {
            console.log(`✅ Created user: ${user.name} (${user.role})`);
            resolve();
          }
        }
      );
    });

    // Add role-specific data
    if (user.role === 'driver') {
      const driverId = uuidv4();
      await new Promise((resolve, reject) => {
        db.run(
          `INSERT OR REPLACE INTO drivers (id, user_id, vehicle_number, badge_number, license_number, ambulance_type) 
           VALUES (?, ?, ?, ?, ?, ?)`,
          [driverId, user.id, 'AMB-001', 'DR-001', 'DL-123456', 'MICU'],
          function(err) {
            if (err) {
              console.error(`Error inserting driver data:`, err);
              reject(err);
            } else {
              console.log(`   📋 Added driver profile for ${user.name}`);
              resolve();
            }
          }
        );
      });
    } else if (user.role === 'medical_expert') {
      const expertId = uuidv4();
      await new Promise((resolve, reject) => {
        db.run(
          `INSERT OR REPLACE INTO medical_experts (id, user_id, specialization, license_number) 
           VALUES (?, ?, ?, ?)`,
          [expertId, user.id, 'Emergency Medicine', 'MED-123456'],
          function(err) {
            if (err) {
              console.error(`Error inserting medical expert data:`, err);
              reject(err);
            } else {
              console.log(`   🩺 Added medical expert profile for ${user.name}`);
              resolve();
            }
          }
        );
      });
    }
  }

  // Add a few more drivers for testing the boosted allocation
  const additionalDrivers = [
    {
      id: uuidv4(),
      name: 'Alex Ambulance',
      email: 'driver2@demo.com',
      phone: '+1-555-0005',
      password: await bcrypt.hash('password123', 10),
      role: 'driver',
      age: 35,
      gender: 'male'
    },
    {
      id: uuidv4(),
      name: 'Lisa Lifesaver',
      email: 'driver3@demo.com',
      phone: '+1-555-0006',
      password: await bcrypt.hash('password123', 10),
      role: 'driver',
      age: 29,
      gender: 'female'
    }
  ];

  for (const driver of additionalDrivers) {
    // Insert user
    await new Promise((resolve, reject) => {
      db.run(
        `INSERT OR REPLACE INTO users (id, name, email, phone, password, role, age, gender) 
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
        [driver.id, driver.name, driver.email, driver.phone, driver.password, driver.role, driver.age, driver.gender],
        function(err) {
          if (err) {
            console.error(`Error inserting driver ${driver.name}:`, err);
            reject(err);
          } else {
            console.log(`✅ Created additional driver: ${driver.name}`);
            resolve();
          }
        }
      );
    });

    // Insert driver profile
    const driverId = uuidv4();
    const ambulanceTypes = ['Basic Unit', 'Pediatric Unit'];
    const ambulanceType = ambulanceTypes[Math.floor(Math.random() * ambulanceTypes.length)];
    
    await new Promise((resolve, reject) => {
      db.run(
        `INSERT OR REPLACE INTO drivers (id, user_id, vehicle_number, badge_number, license_number, ambulance_type) 
         VALUES (?, ?, ?, ?, ?, ?)`,
        [driverId, driver.id, `AMB-00${Math.floor(Math.random() * 9) + 2}`, `DR-00${Math.floor(Math.random() * 9) + 2}`, `DL-${Math.floor(Math.random() * 900000) + 100000}`, ambulanceType],
        function(err) {
          if (err) {
            console.error(`Error inserting driver data:`, err);
            reject(err);
          } else {
            console.log(`   📋 Added driver profile: ${ambulanceType}`);
            resolve();
          }
        }
      );
    });
  }

  console.log('\n🎉 Demo users seeded successfully!');
  console.log('\n📝 Demo Accounts:');
  console.log('Victim: victim@demo.com / password123');
  console.log('Driver: driver@demo.com / password123');
  console.log('Hospital: hospital@demo.com / password123');
  console.log('Expert: expert@demo.com / password123');
  console.log('\n🚀 You can now start the application with: npm run dev');
};

// Run the seeding
seedDemoUsers().then(() => {
  db.close((err) => {
    if (err) {
      console.error('Error closing database:', err);
    } else {
      console.log('Database connection closed.');
    }
    process.exit(0);
  });
}).catch((err) => {
  console.error('Error seeding database:', err);
  process.exit(1);
});