import React from 'react';

import NavigationLink from '../NavigationLink';

import styles from './index.module.css';

const toolbar = props => (
    <nav className={styles.Toolbar}>
        <ul className={styles.NavigationLinks}>
            {props.links.map(link => (
                <NavigationLink key={link.to} to={link.to}>
                    {link.value}
                </NavigationLink>
            ))}
        </ul>
    </nav>
);

export default toolbar;