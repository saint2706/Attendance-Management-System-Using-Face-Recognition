import { Link } from 'react-router-dom';
import React from 'react';

interface ActionCardProps {
    to: string;
    title: string;
    icon: React.ElementType;
    heading: string;
    description: string;
}

/**
 * A reusable card component for navigation actions.
 * Displays an icon, heading, and description within a clickable card.
 *
 * @param {ActionCardProps} props - The properties for the ActionCard component.
 * @param {string} props.to - The destination URL for the link.
 * @param {string} props.title - The title attribute for the link.
 * @param {React.ElementType} props.icon - The icon component to display.
 * @param {string} props.heading - The main heading text.
 * @param {string} props.description - The descriptive text below the heading.
 * @returns {JSX.Element} The rendered ActionCard component.
 */
export const ActionCard = React.memo(({ to, title, icon: Icon, heading, description }: ActionCardProps) => {
    const headingId = React.useId();
    const descId = React.useId();

    return (
        <Link to={to} className="action-card card card-elevated" title={title} aria-labelledby={headingId} aria-describedby={descId}>
            <div className="card-body">
                <Icon width={32} height={32} className="action-icon" aria-hidden="true" />
                <h3 id={headingId}>{heading}</h3>
                <p id={descId} className="text-muted text-sm">
                    {description}
                </p>
            </div>
        </Link>
    );
});
