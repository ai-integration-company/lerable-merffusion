import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import { HomePage } from './pages/Home';
import { BackgroundPage, ParametersPage, ResultsPage } from './pages';

const router = createBrowserRouter([
  {
    path: '/',
    element: <HomePage />,
  },
  {
    path: '/background',
    element: <BackgroundPage />,
  },
  {
    path: '/parameters',
    element: <ParametersPage />,
  },
  {
    path: '/results',
    element: <ResultsPage />,
  },
]);

export function Router() {
  return <RouterProvider router={router} />;
}
