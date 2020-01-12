import React from 'react';

import NavigationLink from '../NavigationLink';

import styles from './index.module.css';

const toolbar = props => {
    const links = [
        { to: 'test1', value: 'Test' },
        { to: 'test2', value: 'Test' },
        { to: 'test3', value: 'Test' },
        { to: 'test4', value: 'Test' }
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