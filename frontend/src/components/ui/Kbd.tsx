import React from 'react';

interface KbdProps {
  children: React.ReactNode;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
}

/**
 * A component to represent keyboard input.
 * Displays styled keyboard shortcut keys.
 * @param {KbdProps} props - The component props.
 * @returns {JSX.Element} The styled keyboard element.
 */
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
