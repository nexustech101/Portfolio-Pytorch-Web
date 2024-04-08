const offset = 1366
const normalizedData = []
const y = []
// window.addEventListener('resize', e => {
//     console.log(window.innerWidth >= 760 ? 1 : 0)
//     normalizedData.push([window.innerWidth / offset, window.innerWidth >= 760 ? 1 : 0]);
//     y.push([window.innerWidth >= 760 ? 1 : 0])
//     if (normalizedData.length > 75) {
//         console.log(normalizedData);
//         // const child = document.createElement('span')
//         // child.innerHTML = normalizedData
//         // normalizedData.forEach(element => {
//         //     document.getElementById('templates').appendChild(element)
//         // })
//     }
// })

const handlePrediction = async (data) => {
    try {
        const url = 'http://127.0.0.1:88/api/v1/heart-disease';
        const payload = { data };
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const result = await response.json();
        return result;
    } catch (error) {
        console.error('Error:', error);
        return null;
    }
};

// Example usage:
const data = [
    [35, 1, 2, 123, 161, 0, 1, 153, 0, -0.1],
    [62, 1, 0, 112, 258, 0, 1, 150, 1, 1.3],
    [43, 1, 3, 122, 0, 0, 0, 120, 0, 0.5]
];

let result;
handlePrediction(data)
    .then(result => {
        const { prediction } = result;
        result = prediction;
        console.log(result)
    })
    .catch(error => {
        console.error('Error:', error);
    });