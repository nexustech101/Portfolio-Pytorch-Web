
// class Boid {
//   constructor(x, y, velocityX, velocityY) {
//     this.x = x;
//     this.y = y;
//     this.velocity = {
//       x: velocityX,
//       y: velocityY
//     }
//     this.maxSpeed = 2;
//     this.maxForce = 0.03 + Math.random() * 0.025;
//   }

//   update(boids) {
//     let alignment = this.align(boids);
//     let cohesion = this.cohere(boids);
//     let separation = this.separate(boids);
//     // let wanderForce = Math.random() <= 0.025 ? this.wander() : { x: 0, y: 0 }

//     this.velocity.x += alignment.x + cohesion.x + separation.x;
//     this.velocity.y += alignment.y + cohesion.y + separation.y;

//     // Limit the speed
//     let speed = Math.sqrt(this.velocity.x ** 2 + this.velocity.y ** 2);
//     if (speed > this.maxSpeed) {
//       this.velocity.x = (this.velocity.x / speed) * this.maxSpeed;
//       this.velocity.y = (this.velocity.y / speed) * this.maxSpeed;
//     }

//     if (drawLines) {
//       boids.forEach(other => {
//         let d = distance(this.x, this.y, other.x, other.y);
//         if (d < 75) { // For distances greater than 100
//           ctx.beginPath();  // Begin a new path for the line
//           ctx.moveTo(this.x, this.y); // Move to the starting point (this boid's position)
//           ctx.lineTo(other.x, other.y); // Draw a line to the other boid's position
//           ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)'; // Set line color and transparency
//           ctx.stroke(); // Render the line
//         }
//       });
//     }

//     // Move boid
//     this.x += this.velocity.x;
//     this.y += this.velocity.y;

//     // Screen wrapping
//     this.x = (this.x + canvas.width) % canvas.width;
//     this.y = (this.y + canvas.height) % canvas.height;
//   }

//   draw(ctx) {
//     ctx.beginPath();
//     ctx.arc(this.x, this.y, 5, 0, 2 * Math.PI);
//     ctx.fill();
//     // ctx.drawImage(this.image, 0, 0, 5, 5)
//   }

//   // Alignment rule implementation
//   align(boids) {
//     let perceptionRadius = 50;
//     let steering = { x: 0, y: 0 };
//     let total = 0;
//     boids.forEach(other => {
//       let d = distance(this.x, this.y, other.x, other.y);
//       if (other !== this && d < perceptionRadius) {
//         steering.x += other.velocity.x;
//         steering.y += other.velocity.y;
//         total++;
//       }
//     });
//     if (total > 0) {
//       steering.x /= total;
//       steering.y /= total;
//       steering.x -= this.velocity.x;
//       steering.y -= this.velocity.y;
//       steering.x += Math.random() * 0.1 - 0.05; // Adding a small random value
//       steering.y += Math.random() * 0.1 - 0.05;
//     }
//     return steering;
//   }

//   // Cohesion rule implementation
//   cohere(boids) {
//     let perceptionRadius = 10;
//     let center = { x: 0, y: 0 };
//     let total = 0;
//     boids.forEach(other => {
//       let d = distance(this.x, this.y, other.x, other.y);
//       if (other !== this && d < perceptionRadius) {
//         center.x += other.x;
//         center.y += other.y;
//         total++;
//       }
//     });
//     if (total > 0) {
//       center.x /= total;
//       center.y /= total;
//       return { x: (center.x - this.x) * 0.05, y: (center.y - this.y) * 0.05 };
//     } else {
//       return { x: 0, y: 0 };
//     }
//   }

//   // Separation rule implementation
//   separate(boids) {
//     let perceptionRadius = 25;
//     let steer = { x: 0, y: 0 };
//     let total = 0;
//     boids.forEach(other => {
//       let d = distance(this.x, this.y, other.x, other.y);
//       if (other !== this && d < perceptionRadius) {
//         let diff = { x: this.x - other.x, y: this.y - other.y };
//         let distanceSquared = d * d;
//         steer.x += diff.x / distanceSquared;
//         steer.y += diff.y / distanceSquared;
//         total++;
//       }
//     });
//     if (total > 0) {
//       steer.x /= total;
//       steer.y /= total;
//     }
//     return steer;
//   }

//   wander() {
//     const angleChange = 0.5; // Adjust as needed for more or less randomness
//     const wanderForce = {
//       x: Math.cos(Math.random() * 2 * Math.PI) * angleChange,
//       y: Math.sin(Math.random() * 2 * Math.PI) * angleChange
//     };
//     return wanderForce;
//   }
// }

// class Flock {
//   constructor() {
//     this.boids = [];
//   }

//   addBoid(boid) {
//     this.boids.push(boid);
//   }

//   run(ctx) {
//     this.boids.forEach(boid => {
//       boid.update(this.boids);
//       boid.draw(ctx);
//     });
//   }
// }

// function distance(x1, y1, x2, y2) {
//   return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
// }

// function drawBackground(image) {
//   ctx.drawImage(image, 0, 0, canvas.width, canvas.height)
// }

// // Setup canvas
// const canvas = document.getElementById('canvas');
// const ctx = canvas.getContext('2d');
// canvas.width = window.innerWidth - 20;
// canvas.height = window.innerHeight - 20;
// canvas.style.borderRadius = '10px'
// const img = new Image(); img.src = '../../img/background2.jpg';
// const image = new Image(); image.src = '../../icons/boid.png';
// let drawLines = false; // Flag to control line drawing
// const toggleButton = document.getElementById('toggleLines');

// toggleButton.addEventListener('click', () => {
//   drawLines = !drawLines; // Toggle the state
//   if (drawLines) {
//     toggleButton.textContent = 'Hide Lines'; // Update button text
//   } else {
//     toggleButton.textContent = 'Show Lines'; // Update button text
//   }
// });

// // Create flock
// const flock = new Flock();
// for (let i = 0; i < 100; i++) {
//   let boid = new Boid(
//     Math.random() * canvas.width,
//     Math.random() * canvas.height,
//     Math.random() * 3 - 1.5,
//     Math.random() * 3 - 1.5
//   );
//   flock.addBoid(boid);
// }

// function animate() {
//   ctx.clearRect(0, 0, canvas.width, canvas.hieght);
//   drawBackground(img);
//   flock.run(ctx);
//   requestAnimationFrame(animate);
// }
// animate();

