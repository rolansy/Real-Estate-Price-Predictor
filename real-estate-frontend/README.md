# Real Estate Price Predictor - Frontend

A modern Angular application for predicting real estate prices in Bangalore using machine learning.

## Features

- 🏠 Real-time price prediction
- 📍 Location-based predictions
- 🎨 Modern UI with Tailwind CSS
- 📱 Responsive design
- ⚡ Fast and efficient

## Tech Stack

- **Framework:** Angular 20
- **Styling:** Tailwind CSS
- **HTTP Client:** Angular HttpClient
- **Deployment:** Vercel

## Getting Started

### Prerequisites

- Node.js (v18 or higher)
- npm or yarn
- Angular CLI

### Installation

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start development server:**
   ```bash
   ng serve
   ```

3. **Open your browser:**
   Navigate to `http://localhost:4200`

### Building for Production

```bash
ng build
```

The build artifacts will be stored in the `dist/` directory.

## Configuration

### API URL

Update the backend API URL in `src/app/services/price-prediction.service.ts`:

```typescript
private apiUrl = 'https://your-backend-url.com'; // Replace with your deployed backend URL
```

## Deployment

### Deploy to Vercel

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/your-username/real-estate-frontend.git
   git push -u origin main
   ```

2. **Deploy to Vercel:**
   - Go to [vercel.com](https://vercel.com)
   - Sign up/Login with GitHub
   - Click "New Project"
   - Import your GitHub repository
   - Configure the project:
     - Framework Preset: Angular
     - Root Directory: `real-estate-frontend`
   - Click "Deploy"

### Environment Variables

You can use Vercel environment variables to configure the API URL:

1. In your Vercel project settings, add:
   - Name: `API_URL`
   - Value: `https://your-backend-url.com`

2. Update the service to use the environment variable (see DEPLOYMENT.md for details)

## Project Structure

```
src/
├── app/
│   ├── components/
│   │   └── price-prediction/
│   │       ├── price-prediction.component.ts
│   │       ├── price-prediction.component.html
│   │       └── price-prediction.component.css
│   ├── services/
│   │   └── price-prediction.service.ts
│   ├── app.ts
│   ├── app.html
│   └── app.config.ts
├── styles.css
└── main.ts
```

## Development

### Code Generation

```bash
# Generate a new component
ng generate component component-name

# Generate a new service
ng generate service service-name
```

### Running Tests

```bash
# Unit tests
ng test

# End-to-end tests
ng e2e
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.
