import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import { HomePage } from './pages/Home';
import { ResultsPage } from './pages';

const router = createBrowserRouter([
  {
    path: '/',
    element: <HomePage />,
  },
  {
    path: '/results',
    element: <ResultsPage />,
  },
]);

export function Router() {
  return <RouterProvider router={router} />;
}
