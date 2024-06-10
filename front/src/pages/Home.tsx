import { useNavigate } from 'react-router-dom';
import { Actions, ProductForm, Stepper } from '@/components';

export const HomePage = () => {
  const navigate = useNavigate();
  const onSubmit = () => {
    navigate('/background');
  };
  return (
    <Stepper>
      <ProductForm onSubmit={onSubmit} actions={<Actions backButtonDisabled />} />
    </Stepper>
  );
};
