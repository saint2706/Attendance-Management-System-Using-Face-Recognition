cat << 'EOF2' > frontend/src/components/ActionCard.tsx
import { Link } from 'react-router-dom';
import React from 'react';

interface ActionCardProps {
    to: string;
    title: string;
    icon: React.ElementType;
    heading: string;
    description: string;
}

export const ActionCard = React.memo(({ to, title, icon: Icon, heading, description }: ActionCardProps) => {
    return (
        <Link to={to} className="action-card card card-elevated" title={title}>
            <div className="card-body">
                <Icon size={32} className="action-icon" aria-hidden="true" />
                <h3>{heading}</h3>
                <p className="text-muted text-sm">
                    {description}
                </p>
            </div>
        </Link>
    );
});
EOF2
