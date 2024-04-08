// // Scene
// const scene = new THREE.Scene();

// // Camera
// const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
// camera.position.z = 5;

// // Renderer
// const renderer = new THREE.WebGLRenderer();
// renderer.setSize(window.innerWidth, window.innerHeight);
// document.body.appendChild(renderer.domElement);

// // Resize listener to make it responsive
// window.addEventListener('resize', () => {
//     renderer.setSize(window.innerWidth, window.innerHeight);
//     camera.aspect = window.innerWidth / window.innerHeight;
//     camera.updateProjectionMatrix();
// });

// // Create a geometry for the cube
// const geometry = new THREE.BoxGeometry();

// // To show vertices as points, we create a PointsMaterial
// const pointsMaterial = new THREE.PointsMaterial({ color: 0xffffff, size: 0.01 });

// // Convert the geometry to a THREE.Points object for vertices
// const points = new THREE.Points(geometry, pointsMaterial);

// // Add points to the scene
// scene.add(points);

// // Create edges for the cube
// const edgesGeometry = new THREE.EdgesGeometry(geometry); // Create an EdgesGeometry from the cube
// const edgesMaterial = new THREE.LineBasicMaterial({ color: 0xffffff }); // Use the same color for simplicity
// const edges = new THREE.LineSegments(edgesGeometry, edgesMaterial); // Create LineSegments for edges

// // Add edges to the scene
// scene.add(edges);

// // Animation loop
// function animate() {
//     requestAnimationFrame(animate);

//     // Rotate the points and edges together
//     points.rotation.x += 0.01;
//     points.rotation.y += 0.01;
//     edges.rotation.x += 0.01; // Ensure the edges rotate with the points
//     edges.rotation.y += 0.01;

//     renderer.render(scene, camera);
// }

// animate();






// // Scene setup
// const scene = new THREE.Scene();

// // Camera setup
// const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
// camera.position.z = 5;

// // Renderer setup
// const renderer = new THREE.WebGLRenderer();
// renderer.setSize(window.innerWidth, window.innerHeight);
// document.body.appendChild(renderer.domElement);

// // Responsive window resize
// window.addEventListener('resize', () => {
//     renderer.setSize(window.innerWidth, window.innerHeight);
//     camera.aspect = window.innerWidth / window.innerHeight;
//     camera.updateProjectionMatrix();
// });

// class Cube {
//     constructor() {
//         const geometry = new THREE.BoxGeometry();
//         const material = new THREE.MeshBasicMaterial({ color: 0xff0000 });
//         this.mesh = new THREE.Mesh(geometry, material);
//     }
// }

// class Cylinder {
//     constructor() {
//         const geometry = new THREE.CylinderGeometry(1, 1, 2, 32);
//         const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
//         this.mesh = new THREE.Mesh(geometry, material);
//     }
// }

// class Pyramid {
//     constructor() {
//         const geometry = new THREE.ConeGeometry(1, 2, 4);
//         const material = new THREE.MeshBasicMaterial({ color: 0x0000ff });
//         this.mesh = new THREE.Mesh(geometry, material);
//     }
// }

// function init() {
//     cube = new Cube();
//     cube.mesh.position.x = -2;
//     scene.add(cube.mesh);

//     cylinder = new Cylinder();
//     scene.add(cylinder.mesh);

//     pyramid = new Pyramid();
//     pyramid.mesh.position.x = 2;
//     scene.add(pyramid.mesh);
// }

// let cube = undefined;
// let cylinder = undefined;
// let pyramid = undefined;


// function animate() {
//     requestAnimationFrame(animate);

//     cube.mesh.rotation.x += 0.00;
//     cube.mesh.rotation.y -= 0.01;

//     cylinder.mesh.rotation.x += 0.01;
//     cylinder.mesh.rotation.y += 0.00;

//     pyramid.mesh.rotation.x += 0.00;
//     pyramid.mesh.rotation.y += 0.01;

//     renderer.render(scene, camera);
// }
// init();
// animate();
