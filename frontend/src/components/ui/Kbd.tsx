import React from 'react';

interface KbdProps {
  children: React.ReactNode;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
}

export const Kbd: React.FC<KbdProps> = ({
  children,
  className = '',
  size = 'md'
}) => {
  return (
    <kbd className={`kbd kbd-${size} ${className}`}>
      {children}
    </kbd>
  );
};
