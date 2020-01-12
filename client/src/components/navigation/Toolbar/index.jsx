import React from 'react';

import NavigationLink from '../NavigationLink';

import styles from './index.module.css';

const toolbar = props => {
    const links = [
        { to: '/', value: 'Home' },
        { to: '/generator', value: 'New font' },
        { to: '/auth', value: 'Authenticate' }
    ];

    return (
        <nav className={styles.Toolbar}>
            <ul className={styles.NavigationLinks}>
                {links.map(link => (
                    <NavigationLink key={link.to} to={link.to}>
                        {link.value}
                    </NavigationLink>
                ))}
            </ul>
        </nav>
    );
};

export default toolbar;