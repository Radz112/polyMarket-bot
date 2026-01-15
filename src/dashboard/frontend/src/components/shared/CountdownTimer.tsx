/**
 * Countdown Timer Component
 */
import React, { useState, useEffect } from 'react';

interface Props {
    expiresAt: string;
    onExpire?: () => void;
}

export const CountdownTimer: React.FC<Props> = ({ expiresAt, onExpire }) => {
    const [timeLeft, setTimeLeft] = useState<number>(0);

    useEffect(() => {
        const updateTimer = () => {
            const now = new Date().getTime();
            const expiry = new Date(expiresAt).getTime();
            const remaining = Math.max(0, Math.floor((expiry - now) / 1000));
            setTimeLeft(remaining);

            if (remaining === 0 && onExpire) {
                onExpire();
            }
        };

        updateTimer();
        const interval = setInterval(updateTimer, 1000);

        return () => clearInterval(interval);
    }, [expiresAt, onExpire]);

    const minutes = Math.floor(timeLeft / 60);
    const seconds = timeLeft % 60;

    const urgencyClass = timeLeft < 30 ? 'critical' : timeLeft < 60 ? 'urgent' : '';

    if (timeLeft === 0) {
        return <span className="countdown expired">Expired</span>;
    }

    return (
        <span className={`countdown ${urgencyClass}`}>
            {minutes}:{seconds.toString().padStart(2, '0')}
        </span>
    );
};

export default CountdownTimer;
